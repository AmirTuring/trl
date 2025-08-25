"""
GRPO Training for Mathematical Reasoning (with per-step reward logging)

This script trains language models using GRPO (Group Relative Policy Optimization)
on mathematical reasoning tasks. The training teaches models to solve math problems
with step-by-step reasoning in a structured format:

- <think>step-by-step reasoning</think>
- \\boxed{final answer}

Reward functions:
1. Format reward: Ensures proper <think></think> and exactly one \\boxed{...}
2. Accuracy reward: Uses math_verify to evaluate mathematical correctness

New in this version:
- Pass@k metrics with **Hugging Face-style cadence**:
  - **Train:** averaged over steps since the last log (windowed by `logging_steps`).
  - **Eval:** aggregated over the entire eval loop at each `eval_steps` point (single log per eval run).
- Per-completion event logging (JSONL) updated **every step** (unchanged)
- Per-step summaries with mean/std per prompt and pass@k (unchanged)
- Separate folder per run: <output_dir>/reward_logs/<run_name>_<timestamp>/ (unchanged)
- Proper WandB metric naming: `train/pass_at_k/{k}` and `eval/pass_at_k/{k}`
- **Only rank 0 reads/prints/logs** - no cross-rank reductions needed
- **No** `num_problems` or other count metrics logged
- **Pass@k computed from JSON files** in `.../step_summaries/step_*.json`
- **Step boundary detection** - flushes step summaries when global step advances

Usage:
    python grpo_math.py --config examples/cli_configs/grpo_math_config.yaml
"""

import logging
import os
import random
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import hashlib
import torch

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from trl import (
    GRPOConfig,
    GRPOTrainer,
    get_peft_config,
    ModelConfig,
    TrlParser,
    DatasetMixtureConfig,
    get_dataset,
    ScriptArguments as TrlScriptArguments,
)
from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig
import wandb

# Import the refactored logging classes
from reward_logging import RewardLogger, LegacyRewardTracker

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################
# Custom dataclasses
########################
@dataclass
class FieldMappingConfig:
    """Configuration for mapping dataset fields to expected format."""
    question_field: str = "question"
    answer_field: str = "answer"

@dataclass
class ScriptArguments(TrlScriptArguments):
    tokenizer_name_or_path: str = None
    dataset_seed: int = 42
    field_mapping: FieldMappingConfig = field(default_factory=FieldMappingConfig)
    validation_field_mapping: FieldMappingConfig = field(default_factory=lambda: None)
    validation_datasets: List[Dict[str, Any]] = field(default_factory=list, metadata={"help": "List of validation dataset configurations"})


########################
# Helper functions
########################

def _extract_boxed_answer(completion: str) -> str:
    """Extract answer from \\boxed{} format, handling nested braces."""
    if "\\boxed{" not in completion:
        return completion
    start_idx = completion.find("\\boxed{") + 7
    brace_count = 1
    end_idx = start_idx
    while end_idx < len(completion) and brace_count > 0:
        if completion[end_idx] == "{":
            brace_count += 1
        elif completion[end_idx] == "}":
            brace_count -= 1
        end_idx += 1
    return completion[start_idx:end_idx - 1].strip()

def _select_for_index(arr, i, G, total_len):
    """Robustly select arr item for completion index i."""
    if isinstance(arr, (list, tuple)):
        if len(arr) == total_len:
            return arr[i]
        if len(arr) * G == total_len:
            return arr[i // max(1, G)]
        if len(arr) == 1:
            return arr[0]
        # fallback
        return arr[min(i, len(arr) - 1)]
    return arr

########################
# Reward functions
########################

def format_reward_func(completions, **kwargs):
    """
    Evaluates completions based on strict format:
      <think> ...nonempty reasoning... </think> followed by exactly one \\boxed{...}

    Rewarding scheme:
      -1.0 : invalid / reward hacking attempt
       0.0 : partially correct but missing constraints
       1.0 : valid format
    """
    rewards = []
    for completion in completions:
        try:
            # Ensure consistency with prompt (prepend <think>)
            completion = "<think>" + completion

            if random.random() < 0.01:
                os.makedirs("completion_samples", exist_ok=True)
                with open("completion_samples/format_completion_samples.txt", "a") as f:
                    f.write("\n\n==============\n")
                    f.write(completion)

            think_pattern = r"^<think>(.*?)</think>"
            think_match = re.search(think_pattern, completion, re.DOTALL)

            if not think_match:
                rewards.append(-1.0)
                continue

            think_content = think_match.group(1).strip()
            after_think = completion[think_match.end():]

            if len(think_content) < 10 or "<think>" in after_think:
                rewards.append(-1.0)
                continue

            boxed_matches = re.findall(r"\\boxed\{.*?\}", completion)
            if len(boxed_matches) != 1 or "\\boxed{" not in after_think:
                rewards.append(-1.0)
                continue

            if len(after_think.strip()) > 4 * len(think_content):
                rewards.append(0.0)
                continue

            rewards.append(1.0)
        except Exception:
            rewards.append(0.0)

    return rewards

def accuracy_reward(completions, target, num_generations: int = 1, **kwargs):
    """
    Reward function that evaluates mathematical correctness using math_verify.
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )

    rewards = []
    G = max(1, int(num_generations))
    total = len(completions)

    for i, completion in enumerate(completions):
        try:
            ground_truth = _select_for_index(target, i, G, total)
            completion_answer = _extract_boxed_answer(completion)
            completion_answer = "\\boxed{" + completion_answer + "}"
            ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"

            if ground_truth_boxed.strip().lower() == completion_answer.strip().lower():
                reward = 1.0
            else:
                score, _ = verify_func([ground_truth_boxed], [completion_answer])
                reward = float(score)
        except Exception:
            reward = 0.0
        rewards.append(reward)

    return rewards

########################
# Trainer
########################

class IndexedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = tokenizer
        self._is_evaluation_mode = False  # Track evaluation mode
        self._last_global_step = -1  # Track step boundaries

        try:
            if getattr(self.args, "gradient_checkpointing", False):
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
                if hasattr(self.model, "config"):
                    self.model.config.use_cache = False
        except Exception:
            pass

        try:
            trainable, total = sum(p.numel() for p in self.model.parameters() if p.requires_grad), sum(p.numel() for p in self.model.parameters())
            self.accelerator.print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        except Exception:
            pass

        # --- Instantiate reward loggers ---
        self.reward_logger = None
        self.legacy_tracker = None
        try:
            is_main = self.accelerator.is_main_process
            base_dir = os.path.join(self.args.output_dir, "reward_logs")
            run_name = getattr(self.args, "run_name", "grpo-run")

            num_generations = getattr(self.args, "num_generations", 1)
            k_values = self._k_grid(num_generations)

            self.reward_logger = RewardLogger(
                base_dir=base_dir,
                run_name=run_name,
                is_main=is_main,
                num_generations=num_generations,
                k_defaults=tuple(k_values),
            )
            self.legacy_tracker = LegacyRewardTracker(is_main=is_main)
        except Exception as e:
            self.accelerator.print(f"Logger instantiation failed: {e}")

        # Track evaluation mode
        self._is_evaluation_mode = False

        # Accumulate data for pass@k calculation across logging steps (HF-style)
        self._step_correct_answers = []
        self._step_problem_ids = []
        self._last_logged_step = -1

        # Accumulate pass@k metrics across logging window
        self._logging_window_passk_data = []  # List of pass@k dicts from each step
        self._logging_window_start_step = 0

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Override to calculate rewards and compute pass@k metrics."""
        # Track evaluation mode
        self._is_evaluation_mode = not self.model.training
        if self.accelerator.is_main_process:
            logger.debug(f"_CALCULATE_REWARDS: Called at step {self.state.global_step}, model.training={self.model.training}, "
                        f"eval_mode={self._is_evaluation_mode}, current data size: {len(self._step_correct_answers)}")

        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state
        reward_kwargs["num_generations"] = getattr(self.args, "num_generations", 1)

        for i, (reward_func, _, reward_func_name) in enumerate(zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)):
            output_reward_func = reward_func(prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs)
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            logger.warning(f"All reward functions returned None for a sample. Prompt: {prompts[nan_row_idx][:100]}...")

        # Gather across processes (if needed)
        if hasattr(self.accelerator, "gather"):
            rewards_per_func = self.accelerator.gather(rewards_per_func)
            # Also gather problem_ids to ensure consistency across processes
            problem_ids = reward_kwargs.get("problem_id", [])
            if problem_ids:
                # Gather problem IDs - convert to tensor first for proper gathering
                problem_ids_tensor = torch.tensor(problem_ids, device=self.accelerator.device)
                gathered_problem_ids_tensor = self.accelerator.gather(problem_ids_tensor)
                gathered_problem_ids = gathered_problem_ids_tensor.cpu().tolist()
                reward_kwargs["problem_id"] = gathered_problem_ids

        # Accumulate data for pass@k calculation (main process only)
        if self.accelerator.is_main_process:
            try:
                self._accumulate_step_data(reward_kwargs, rewards_per_func, self.reward_func_names)
            except Exception as e:
                self.accelerator.print(f"Data accumulation failed: {e}")

        return rewards_per_func

    def _accumulate_step_data(self, reward_kwargs, rewards_per_func, reward_func_names):
        """Accumulate data from all batches within a step for pass@k calculation."""
        try:
            logger.debug(f"_ACCUMULATE_DATA: Called with eval_mode={self._is_evaluation_mode}, "
                        f"batch_size={len(reward_kwargs.get('problem_id', []))}")

            # Find accuracy reward index
            acc_idx = reward_func_names.index("accuracy_reward")

            # Get accuracy rewards and determine correct answers
            accuracy_rewards = rewards_per_func[:, acc_idx].cpu().tolist()
            correct_answers = [1.0 if acc >= 0.999 else 0.0 for acc in accuracy_rewards]

            # Get problem IDs
            problem_ids = reward_kwargs.get("problem_id", [])

            logger.debug(f"_ACCUMULATE_DATA: Processing {len(correct_answers)} answers, {len(problem_ids)} problem IDs, "
                        f"accuracy rewards mean: {sum(accuracy_rewards)/len(accuracy_rewards):.4f}")

            # Accumulate data
            self._step_correct_answers.extend(correct_answers)
            self._step_problem_ids.extend(problem_ids)

            logger.debug(f"_ACCUMULATE_DATA: Total accumulated - {len(self._step_correct_answers)} answers, "
                        f"{len(self._step_problem_ids)} problem IDs")

        except Exception as e:
            self.accelerator.print(f"Data accumulation failed: {e}")

    def _calculate_pass_at_k_once_per_step(self):
        """Calculate pass@k metrics once per step using correct combinatorial formula."""
        try:
            logger.info(f"CALCULATE_PASSK: Called with eval_mode={self._is_evaluation_mode}, "
                       f"step_answers={len(self._step_correct_answers)}, step_ids={len(self._step_problem_ids)}")

            if not self._step_correct_answers or not self._step_problem_ids:
                logger.info(f"CALCULATE_PASSK: No step data available - answers: {len(self._step_correct_answers)}, "
                           f"ids: {len(self._step_problem_ids)}")
                return {}

            num_generations = getattr(self.args, "num_generations", 1)

            # Group completions per problem
            problem_groups = []
            if self._step_problem_ids:
                from collections import defaultdict
                grouped = defaultdict(list)
                for pid, correct in zip(self._step_problem_ids, self._step_correct_answers):
                    grouped[int(pid)].append(int(correct))
                problem_groups = list(grouped.values())
                logger.info(f"CALCULATE_PASSK: Grouped by problem IDs, got {len(problem_groups)} problem groups")
            else:
                # Fallback: chunk by num_generations
                cg = int(num_generations)
                for i in range(0, len(self._step_correct_answers), cg):
                    problem_groups.append(self._step_correct_answers[i:i+cg])
                logger.info(f"CALCULATE_PASSK: Used fallback chunking, got {len(problem_groups)} problem groups")

            if not problem_groups:
                logger.info(f"CALCULATE_PASSK: No problem groups created")
                return {}

            # Debug: check problem group sizes
            for idx, corrects in enumerate(problem_groups):
                logger.debug(f"Problem idx {idx}: {len(corrects)} completions, {sum(corrects)} correct")

            # Calculate k values
            k_values = self._k_grid(num_generations)
            logger.info(f"Using k_values: {k_values} for num_generations={num_generations}")

            # Validate that we have enough data for meaningful pass@k calculation
            expected_completions_per_problem = num_generations
            valid_problems = [pg for pg in problem_groups if len(pg) == expected_completions_per_problem]

            if len(valid_problems) != len(problem_groups):
                logger.warning(f"CALCULATE_PASSK: Found {len(problem_groups)} total problems, "
                              f"but only {len(valid_problems)} have exactly {expected_completions_per_problem} completions. "
                              f"Using only valid problems for pass@k calculation.")

            if not valid_problems:
                logger.warning(f"CALCULATE_PASSK: No valid problems found for pass@k calculation")
                return {}

            # Calculate pass@k using correct combinatorial formula
            pass_at_k_metrics = {}
            num_problems = len(valid_problems)

            logger.info(f"Calculating pass@k for {num_problems} problems with {num_generations} generations each")
            logger.info(f"Total accumulated data: {len(self._step_correct_answers)} answers, {len(self._step_problem_ids)} problem IDs")

            if num_problems > 0:
                for k in k_values:
                    pass_counts = []
                    for problem_corrects in valid_problems:
                        # Each problem should have exactly num_generations completions
                        num_correct = int(sum(problem_corrects))  # Convert to int for comb()
                        num_total = len(problem_corrects)

                        logger.debug(f"Problem: {num_correct}/{num_total} correct, k={k}")

                        # Use correct combinatorial formula: pass@k = 1 - C(num_total - num_correct, k) / C(num_total, k)
                        if k <= num_total:
                            from math import comb
                            n_minus_c = num_total - num_correct
                            numerator = comb(n_minus_c, k)
                            denominator = comb(num_total, k)
                            pass_rate = 1.0 - (numerator / denominator) if denominator > 0 else 0.0
                        else:
                            pass_rate = 0.0

                        logger.debug(f"  k={k}: num_correct={num_correct}, num_total={num_total}, pass_rate={pass_rate}")
                        pass_counts.append(pass_rate)

                    if pass_counts:
                        avg_pass_at_k = sum(pass_counts) / len(pass_counts)
                        logger.debug(f"  k={k}: avg_pass_at_k={avg_pass_at_k} from {len(pass_counts)} problems")
                        # Round to 4 decimal places for higher accuracy
                        pass_at_k_metrics[k] = round(avg_pass_at_k, 4)

            # Clear accumulated data (only during training, not during evaluation)
            # During evaluation, we want to keep data for final calculation in evaluate() method
            if not self._is_evaluation_mode:
                self._step_correct_answers.clear()
                self._step_problem_ids.clear()
                logger.info(f"CALCULATE_PASSK: Cleared step data after training calculation")
            else:
                logger.info(f"CALCULATE_PASSK: Keeping step data for evaluation final calculation")

            return pass_at_k_metrics

        except Exception as e:
            self.accelerator.print(f"Pass@k calculation failed: {e}")
            return {}

    def _calculate_window_average_passk(self):
        """Calculate average pass@k metrics across the logging window."""
        try:
            if not self._logging_window_passk_data:
                return {}

            # Get all unique k values across all steps
            all_k_values = set()
            for step_data in self._logging_window_passk_data:
                all_k_values.update(step_data.keys())

            if not all_k_values:
                return {}

            # Calculate average for each k value
            averaged_metrics = {}
            for k in sorted(all_k_values):
                k_values = []
                for step_data in self._logging_window_passk_data:
                    if k in step_data:
                        k_values.append(step_data[k])

                if k_values:
                    # Average the pass@k rates across steps in the window
                    avg_value = sum(k_values) / len(k_values)
                    logger.debug(f"Window avg for {k}: {avg_value} from values {k_values}")
                    averaged_metrics[k] = avg_value

            return averaged_metrics

        except Exception as e:
            self.accelerator.print(f"Window average pass@k calculation failed: {e}")
            return {}

    def _log_pass_at_k_to_wandb(self):
        """Log accumulated pass@k metrics to WandB at configured logging steps using HF-style accumulation."""
        try:
            current_step = int(self.state.global_step)

            # Only log if we haven't logged for this step yet (train path)
            if current_step <= self._last_logged_step:
                return

            # Always calculate pass@k for current step and accumulate
            pass_at_k_metrics = self._calculate_pass_at_k_once_per_step()
            if pass_at_k_metrics:
                self._logging_window_passk_data.append(pass_at_k_metrics)

            # Check if we should log based on logging configuration
            if not self._should_log_now(self._is_evaluation_mode):
                # Continue accumulating data but don't log yet
                return

            # Log averaged pass@k metrics across the logging window
            if self._logging_window_passk_data:
                # Calculate average pass@k across all steps in the window
                averaged_passk = self._calculate_window_average_passk()

                if averaged_passk:
                    # Use evaluation mode flag for proper metric naming
                    prefix = "eval/" if self._is_evaluation_mode else "train/"

                    # Log to WandB with proper metric structure
                    if wandb.run:
                        wandb_metrics = {}
                        for k, v in averaged_passk.items():
                            wandb_metrics[f"{prefix}pass@{k}"] = v
                        wandb.log(wandb_metrics)

                        # Format the metrics with more decimal places for display
                        formatted_metrics = {str(k): f"{v:.4f}" for k, v in averaged_passk.items()}
                        logger.info(f"Step {current_step} ({prefix.strip('/')}) avg pass@k over {len(self._logging_window_passk_data)} steps: {formatted_metrics}")

                    # Clear accumulated data for next logging window
                    self._logging_window_passk_data.clear()
                    self._logging_window_start_step = current_step

                # Update last logged step
                self._last_logged_step = current_step

        except Exception as e:
            self.accelerator.print(f"Pass@k logging failed: {e}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to log pass@k metrics at the end of each step."""
        # Call parent training step
        result = super().training_step(model, inputs, num_items_in_batch)

        # Log pass@k metrics at logging steps (main process only)
        if self.accelerator.is_main_process:
            try:
                logger.info(f"TRAINING_STEP: About to log pass@k, eval_mode={self._is_evaluation_mode}, "
                          f"step_data_size={len(self._step_correct_answers)}")
                self._log_pass_at_k_to_wandb()
            except Exception as e:
                self.accelerator.print(f"Pass@k logging failed: {e}")

        return result

    def evaluation_step(self, model, inputs, num_items_in_batch=None):
        """Override evaluation_step to accumulate pass@k metrics during evaluation."""
        # Call parent evaluation step
        result = super().evaluation_step(model, inputs, num_items_in_batch)

        # During eval we only accumulate raw data; we calculate metrics once after the eval loop
        if self.accelerator.is_main_process:
            try:
                logger.info(f"EVALUATION_STEP: Called at global step {self.state.global_step}, "
                           f"eval_mode={self._is_evaluation_mode}, "
                           f"step_data_size={len(self._step_correct_answers)}")

                # Just log that we're accumulating data, don't calculate metrics yet
                logger.info(f"EVAL ACCUMULATION: Accumulated data - {len(self._step_correct_answers)} answers, "
                           f"{len(self._step_problem_ids)} problem IDs")
            except Exception as e:
                self.accelerator.print(f"Eval data accumulation failed: {e}")

        return result

    def evaluate(self, *args, **kwargs):
        """Run the normal evaluation loop, then log aggregated pass@k once."""
        logger.info(f"EVALUATE METHOD: Called at step {self.state.global_step}")

        # Clear any stale accumulators
        self._logging_window_passk_data.clear()
        self._step_correct_answers.clear()
        self._step_problem_ids.clear()

        # Set evaluation mode for proper metric logging
        self._is_evaluation_mode = True

        out = super().evaluate(*args, **kwargs)

        if self.accelerator.is_main_process:
            # Calculate pass@k from accumulated step data
            if self._step_correct_answers and self._step_problem_ids:
                final_metrics = self._calculate_pass_at_k_once_per_step()

                if final_metrics and wandb.run:
                    wandb_metrics = {f"eval/pass@{k}": v for k, v in final_metrics.items()}

                    # Log evaluation metrics to current WandB step using commit=False
                    # This prevents creating a new step and avoids monotonicity issues
                    wandb.log(wandb_metrics, commit=False)

                    # Log to console as well
                    formatted_metrics = {f"pass@{k}": f"{v:.4f}" for k, v in final_metrics.items()}
                    logger.info(f"Eval pass@k at step {self.state.global_step}: {formatted_metrics}")
                else:
                    logger.info("EVALUATE METHOD: No final metrics calculated or no wandb run")
            else:
                logger.warning("EVALUATE METHOD: No step data available for pass@k calculation")

        # Reset window after logging and reset evaluation mode
        self._logging_window_passk_data.clear()
        self._is_evaluation_mode = False

        # Clear step data after evaluation to prepare for next training step
        logger.info(f"EVALUATE METHOD: Clearing step data after evaluation - had {len(self._step_correct_answers)} answers")
        self._step_correct_answers.clear()
        self._step_problem_ids.clear()

        return out

    def _should_log_now(self, is_eval: bool) -> bool:
        """Check if we should log metrics based on logging configuration."""
        step = int(self.state.global_step)
        if is_eval:
            eval_strategy = getattr(self.args, "eval_strategy", "no")
            eval_steps = getattr(self.args, "eval_steps", None)
            if eval_strategy == "steps" and eval_steps:
                return (step % eval_steps == 0) or (step == 0)
            return eval_strategy != "no"
        else:
            logging_strategy = getattr(self.args, "logging_strategy", "steps")
            logging_steps = getattr(self.args, "logging_steps", None)
            logging_first_step = getattr(self.args, "logging_first_step", False)
            if logging_strategy == "steps" and logging_steps:
                if step % logging_steps == 0:
                    return True
                if logging_first_step and step == 0:
                    return True
                return False
            return True

    def finalize_pending_eval_passk(self):
        """Flush any pending evaluation pass@k metrics at the end of training."""
        try:
            if self.accelerator.is_main_process and self._logging_window_passk_data:
                # Force logging of any pending evaluation metrics
                current_step = int(self.state.global_step)

                # Calculate average pass@k across all steps in the window
                averaged_passk = self._calculate_window_average_passk()

                if averaged_passk:
                    # Use evaluation mode flag for proper metric naming
                    prefix = "eval/" if self._is_evaluation_mode else "train/"

                    # Log to WandB with proper metric structure
                    if wandb.run:
                        wandb_metrics = {}
                        for k, v in averaged_passk.items():
                            wandb_metrics[f"{prefix}pass@{k}"] = v
                        wandb.log(wandb_metrics)

                        # Format the metrics with more decimal places for display
                        formatted_metrics = {str(k): f"{v:.4f}" for k, v in averaged_passk.items()}
                        logger.info(f"Step {current_step} (final {prefix.strip('/')}) avg pass@k over {len(self._logging_window_passk_data)} steps: {formatted_metrics}")

                    # Clear accumulated data for next logging window
                    self._logging_window_passk_data.clear()
                    self._logging_window_start_step = current_step

        except Exception as e:
            self.accelerator.print(f"Finalize eval pass@k failed: {e}")

    def _k_grid(self, num_generations: int):
        """Generate k values as powers of 2 until num_generations / 2."""
        k_values = [1]
        current_k = 2
        max_k = max(1, num_generations // 2)  # Powers of 2 until num_generations / 2
        while current_k <= max_k:
            k_values.append(current_k)
            current_k *= 2
        return sorted(list(set(k_values)))


########################
# Utilities
########################

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    return last_checkpoint

########################
# Main training function
########################

def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig, dataset_args: DatasetMixtureConfig
):
    #########################
    # Initialize WandB (main process only)
    #########################
    is_main_process = training_args.local_rank in [-1, 0]

    # Disable W&B early on non-main ranks to avoid accidental init
    if not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"

    if hasattr(training_args, "report_to") and "wandb" in training_args.report_to and is_main_process:
        wandb_config = {
            "model": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs,
            "beta": training_args.beta,
            "max_prompt_length": training_args.max_prompt_length,
            "max_completion_length": training_args.max_completion_length,
            "num_generations": training_args.num_generations,
            "total_batch_size": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size,
            "world_size": training_args.world_size,
        }

        run_name = getattr(training_args, "run_name", "grpo-math-training")

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project="grpo-math-training",
            name=run_name,
            config=wandb_config,
            tags=["grpo", "math", "reasoning"],
            settings=wandb.Settings(start_method="fork"),
        )
        logger.info(f"W&B initialized on main process (world_size: {training_args.world_size})")
    else:
        logger.info(f"W&B disabled on rank {training_args.local_rank}")

    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############

    logger.info("Loading dataset using YAML configuration...")
    try:
        # Load training dataset using TRL's DatasetMixtureConfig
        if dataset_args.datasets:
            dataset = get_dataset(dataset_args)
        else:
            raise ValueError("No datasets specified in configuration. Please provide a YAML config with datasets.")

        # Get training dataset
        train_dataset = dataset.get(script_args.dataset_train_split)
        if train_dataset is None:
            available_splits = list(dataset.keys())
            raise ValueError(f"No dataset found for split '{script_args.dataset_train_split}'. Available splits: {available_splits}")

        logger.info(f"Training dataset size: {len(train_dataset)} (using split: '{script_args.dataset_train_split}')")

        # Get validation dataset if available and evaluation is enabled
        eval_dataset = None
        if training_args.eval_strategy != "no":
            # Check if separate validation datasets are specified
            if script_args.validation_datasets:
                logger.info("Loading separate validation datasets...")
                from trl.scripts.utils import DatasetConfig
                validation_dataset_configs = []
                for val_dataset in script_args.validation_datasets:
                    validation_dataset_configs.append(DatasetConfig(**val_dataset))

                validation_mixture_config = DatasetMixtureConfig(
                    datasets=validation_dataset_configs,
                    streaming=dataset_args.streaming,
                    test_split_size=None,
                )
                validation_dataset_dict = get_dataset(validation_mixture_config)
                eval_dataset = validation_dataset_dict.get(script_args.dataset_test_split)
                if eval_dataset is not None:
                    logger.info(f"Validation dataset size: {len(eval_dataset)} (using separate validation datasets)")
                else:
                    available_splits = list(validation_dataset_dict.keys())
                    logger.info(f"No validation dataset found for split '{script_args.dataset_test_split}'. Available splits: {available_splits}")
                    if available_splits:
                        eval_dataset = validation_dataset_dict[available_splits[0]]
                        logger.info(f"Using first available split '{available_splits[0]}' for validation. Size: {len(eval_dataset)}")
            else:
                eval_dataset = dataset.get(script_args.dataset_test_split)
                if eval_dataset is not None:
                    logger.info(f"Validation dataset size: {len(eval_dataset)} (using split: '{script_args.dataset_test_split}')")
                else:
                    available_splits = list(dataset.keys())
                    logger.info(f"No validation dataset found for split '{script_args.dataset_test_split}'. Available splits: {available_splits}")
                    logger.info("Training will proceed without validation.")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Generate prompt for mathematical reasoning
    def generate_math_prompt(question, answer):
        """Generate prompt with step-by-step thinking format."""
        try:
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
                },
                {
                    "role": "user",
                    "content": f"{question}\n\nThink step-by-step inside <think>...</think> tags, then give your final answer inside \\boxed{{}}.",
                },
                {"role": "assistant", "content": "<think>"},
            ]

            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=True)
            return {"prompt": prompt, "target": answer}
        except Exception as e:
            logger.error(f"Error generating math prompt: {e}")
            # Fallback to simple format
            return {"prompt": f"Solve: {question}\nThink step by step.\n<think>", "target": answer}

    # Get field mapping configuration for training
    train_question_field = script_args.field_mapping.get("question_field")
    train_answer_field = script_args.field_mapping.get("answer_field")

    logger.info(f"Using training field mapping - Question: '{train_question_field}', Answer: '{train_answer_field}'")

    # Get field mapping configuration for validation (fallback to training mapping if not specified)
    if script_args.validation_field_mapping is not None:
        eval_question_field = script_args.validation_field_mapping.get("question_field")
        eval_answer_field = script_args.validation_field_mapping.get("answer_field")
        logger.info(f"Using validation field mapping - Question: '{eval_question_field}', Answer: '{eval_answer_field}'")
    else:
        eval_question_field = train_question_field
        eval_answer_field = train_answer_field
        logger.info("Using same field mapping for validation as training")

    # Convert dataset to prompt format + include problem_id
    def extract_and_format_train(row, idx):
        """Extract question and answer using training field mapping. Add problem_id + raw fields."""
        question = row[train_question_field]
        answer = row[train_answer_field]
        prompt_data = generate_math_prompt(question, answer)
        prompt_data["problem_id"] = int(row.get("id", idx))
        prompt_data["raw_question"] = question
        prompt_data["raw_answer"] = answer
        return prompt_data

    def extract_and_format_eval(row, idx):
        """Extract question and answer using validation field mapping. Add problem_id + raw fields."""
        question = row[eval_question_field]
        answer = row[eval_answer_field]
        prompt_data = generate_math_prompt(question, answer)
        prompt_data["problem_id"] = int(row.get("id", idx))
        prompt_data["raw_question"] = question
        prompt_data["raw_answer"] = answer
        return prompt_data

    logger.info("Converting dataset to prompt format...")
    train_dataset = train_dataset.map(extract_and_format_train, with_indices=True, desc="Processing train dataset")

    # Process validation dataset if available
    if eval_dataset is not None:
        logger.info("Converting validation dataset to prompt format...")
        eval_dataset = eval_dataset.map(extract_and_format_eval, with_indices=True, desc="Processing validation dataset")

    # Verify the dataset structure
    logger.info("Verifying dataset structure...")
    sample = train_dataset[0]
    required_fields = ["prompt", "target", "problem_id"]
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        logger.error(f"Missing required fields in dataset: {missing_fields}")
        logger.error(f"Available fields: {list(sample.keys())}")
        raise ValueError(f"Dataset missing required fields: {missing_fields}")

    logger.info(f"✓ Dataset validation passed. Sample prompt length: {len(sample['prompt'])}")
    if len(sample["prompt"]) > 1000:  # Show truncated version for long prompts
        logger.info(f"Sample prompt (first 200 chars): {sample['prompt'][:200]}...")
    else:
        logger.info(f"Sample prompt: {sample['prompt']}")
    logger.info(f"Sample target: {sample['target']}")
    logger.info(f"Sample problem_id: {sample['problem_id']}")

    reward_functions = [format_reward_func, accuracy_reward]
    trainer = IndexedGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        tokenizer=tokenizer,
    )
    if getattr(trainer, "reward_logger", None):
        trainer.reward_logger.write_meta(model_args, training_args, dataset_args)

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    train_info = f"{training_args.max_steps} steps" if training_args.max_steps > 0 else f"{training_args.num_train_epochs} epochs"
    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {train_info} ***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and log final stats
    ##################################
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model and tokenizer saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo"]})
        # Save and log final legacy reward statistics
        if trainer.legacy_tracker:
            logger.info("*** Saving final reward statistics per problem (legacy snapshots) ***")
            trainer.legacy_tracker.save_statistics(step=trainer.state.global_step)
            trainer.legacy_tracker.print_summary()

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(commit_message=f"GRPO training checkpoint - Step {trainer.state.global_step}")

    # Flush any pending eval pass@k (e.g., if training ended immediately after eval)
    trainer.finalize_pending_eval_passk()

    # Finalize all logging
    if trainer.accelerator.is_main_process:
        if getattr(trainer, "reward_logger", None):
            trainer.reward_logger.flush_step(int(trainer.state.global_step))
            trainer.reward_logger.finalize()
            logger.info(f"Modern reward logs written to: {trainer.reward_logger.run_dir}")
        if wandb.run:
            wandb.finish()
            logger.info("W&B logging finished")

    logger.info("*** All tasks complete! ***")

########################
# CLI entry
########################

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()
    grpo_function(model_args, script_args, training_args, dataset_args)

if __name__ == "__main__":
    main()
