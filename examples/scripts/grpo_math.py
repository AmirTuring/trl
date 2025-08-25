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
- Multi-GPU correctness via **SUM reduction** (no biased averaging)
- W&B init/login only on main process; non-main ranks **disabled early**
- **No** `num_problems` or other count metrics logged

Usage:
    python grpo_math_passk_wandb_fixed.py --config examples/cli_configs/grpo_math_config.yaml
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
    logger.info(f"format_reward_func called with {len(completions)} completions")

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
    logger.info(f"accuracy_reward called with {len(completions)} completions")

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

        # --- Pass@k accumulators (HF-style behavior) ---
        self._train_pk_num_local: Dict[int, int] = {}
        self._train_pk_den_local: Dict[int, int] = {}
        self._eval_pk_num_local: Dict[int, int] = {}
        self._eval_pk_den_local: Dict[int, int] = {}
        self._eval_accumulating: bool = False
        self._eval_start_step: int | None = None

    def _prepare_inputs(self, inputs):
        """Override to track evaluation mode."""
        # Set evaluation mode flag based on model training state
        self._is_evaluation_mode = not self.model.training
        return super()._prepare_inputs(inputs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Override to calculate rewards and stream events per completion."""
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

        # --- Per-batch event logging (main process only) ---
        if self.accelerator.is_main_process:
            try:
                if self.reward_logger:
                    self._record_reward_events(prompts, completions, reward_kwargs, rewards_per_func, self.reward_func_names)
                    self.reward_logger.flush_step(int(self.state.global_step))
                if self.legacy_tracker:
                    self._update_legacy_tracker(reward_kwargs, rewards_per_func, self.reward_func_names)
            except Exception as e:
                self.accelerator.print(f"Reward event logging/tracking failed: {e}")

        # --- Pass@k accumulation/logging (must run on ALL ranks for proper reduction) ---
        try:
            self._accumulate_and_maybe_log_passk(reward_kwargs, rewards_per_func, self.reward_func_names)
        except Exception as e:
            self.accelerator.print(f"Pass@k accumulation/logging failed: {e}")

        return rewards_per_func

    def _k_grid(self, num_generations: int):
        k_values = [1]
        current_k = 2
        while current_k <= num_generations:
            k_values.append(current_k)
            current_k *= 2
        if num_generations not in k_values:
            k_values.append(num_generations)
        return sorted(list(set(k_values)))

    def _should_log_now(self, is_eval: bool) -> bool:
        """Gate W&B logging by HF/TRL logging/eval strategies/steps."""
        step = int(self.state.global_step)
        if is_eval:
            eval_strategy = getattr(self.args, "eval_strategy", "no")
            eval_steps = getattr(self.args, "eval_steps", None)
            if eval_strategy == "steps" and eval_steps:
                return (step % eval_steps == 0) or (step == 0)
            return eval_strategy != "no"  # if 'epoch' or others, allow
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
            return True  # e.g., 'epoch'

    def _accumulate_and_maybe_log_passk(self, reward_kwargs, rewards_per_func, reward_func_names):
        """HF-style behavior: accumulate per-batch local counts, then log windowed train averages
        and full-eval aggregates at the appropriate times.
        """
        # Determine mode
        is_eval = self._is_evaluation_mode

        # Build local correctness for this batch
        acc_idx = reward_func_names.index("accuracy_reward")
        accuracy_rewards = rewards_per_func[:, acc_idx]
        correct = (accuracy_rewards >= 0.999).to(torch.int32)

        problem_ids = reward_kwargs.get("problem_id", [])
        if not problem_ids:
            return
        device = self.accelerator.device
        pid_tensor = torch.tensor(problem_ids, device=device, dtype=torch.long)
        num_generations = int(reward_kwargs.get("num_generations", getattr(self.args, "num_generations", 1)))
        k_values = self._k_grid(num_generations)

        # Compute local numerators/denominators for this batch
        local_num: Dict[int, int] = {k: 0 for k in k_values}
        local_den: Dict[int, int] = {k: 0 for k in k_values}
        passed_by_pid: Dict[int, List[int]] = {}
        for p, c in zip(pid_tensor.tolist(), correct.tolist()):
            passed_by_pid.setdefault(int(p), []).append(int(c))
        for k in k_values:
            for arr in passed_by_pid.values():
                if len(arr) >= k:
                    local_den[k] += 1
                    local_num[k] += 1 if sum(arr[:k]) > 0 else 0

        # Helper to add dicts
        def add_into(dst: Dict[int, int], src: Dict[int, int]):
            for kk, vv in src.items():
                dst[kk] = dst.get(kk, 0) + int(vv)

        # --- EVAL path: accumulate across entire eval loop; log once afterwards ---
        if is_eval:
            if not self._eval_accumulating:
                self._eval_accumulating = True
                self._eval_start_step = int(self.state.global_step)
            add_into(self._eval_pk_num_local, local_num)
            add_into(self._eval_pk_den_local, local_den)
            return

        # --- TRAIN path ---
        # If we have pending eval accumulations, finalize and log them now
        if self._eval_accumulating:
            self._finalize_and_log_eval_passk()

        # Always accumulate train counts; we will log on cadence
        add_into(self._train_pk_num_local, local_num)
        add_into(self._train_pk_den_local, local_den)

        if not self._should_log_now(is_eval=False):
            return

        # Time to emit a windowed train log
        self._finalize_and_log_train_passk()

    def _finalize_and_log_train_passk(self):
        device = self.accelerator.device
        # Reduce sums across processes
        reduced_rates: Dict[int, float] = {}
        for k in list(self._train_pk_den_local.keys() | self._train_pk_num_local.keys()):
            n = torch.tensor(self._train_pk_num_local.get(k, 0), device=device, dtype=torch.int64)
            d = torch.tensor(self._train_pk_den_local.get(k, 0), device=device, dtype=torch.int64)
            n = self.accelerator.reduce(n, reduction="sum")
            d = self.accelerator.reduce(d, reduction="sum")
            n_i, d_i = int(n.item()), int(d.item())
            reduced_rates[k] = (n_i / d_i) if d_i else 0.0
        # Reset accumulators for next window
        self._train_pk_num_local.clear()
        self._train_pk_den_local.clear()
        # Log on main only, at the current global step
        if self.accelerator.is_main_process and wandb.run:
            step = int(self.state.global_step)
            metrics = {f"train/pass_at_k/{k}": v for k, v in reduced_rates.items()}
            wandb.log(metrics, step=step)

    def _finalize_and_log_eval_passk(self):
        device = self.accelerator.device
        reduced_rates: Dict[int, float] = {}
        for k in list(self._eval_pk_den_local.keys() | self._eval_pk_num_local.keys()):
            n = torch.tensor(self._eval_pk_num_local.get(k, 0), device=device, dtype=torch.int64)
            d = torch.tensor(self._eval_pk_den_local.get(k, 0), device=device, dtype=torch.int64)
            n = self.accelerator.reduce(n, reduction="sum")
            d = self.accelerator.reduce(d, reduction="sum")
            n_i, d_i = int(n.item()), int(d.item())
            reduced_rates[k] = (n_i / d_i) if d_i else 0.0
        # Capture and reset state BEFORE logging
        eval_step = self._eval_start_step if self._eval_start_step is not None else int(self.state.global_step)
        self._eval_pk_num_local.clear()
        self._eval_pk_den_local.clear()
        self._eval_accumulating = False
        self._eval_start_step = None
        # Log on main only, at the eval step
        if self.accelerator.is_main_process and wandb.run:
            metrics = {f"eval/pass_at_k/{k}": v for k, v in reduced_rates.items()}
            wandb.log(metrics, step=eval_step)

    def _update_legacy_tracker(self, reward_kwargs, rewards_per_func, reward_func_names):
        """Update the legacy reward tracker with a batch of rewards."""
        name_map = {"format_reward_func": "format_reward", "accuracy_reward": "accuracy_reward"}
        num_gens = int(reward_kwargs.get("num_generations", 1))
        problem_ids = reward_kwargs.get("problem_id", [])

        for i, name in enumerate(reward_func_names):
            tracker_name = name_map.get(name)
            if tracker_name:
                max_len = len(problem_ids) * max(1, num_gens)
                rewards = rewards_per_func[:max_len, i].detach().cpu().tolist()
                self.legacy_tracker.update_batch(
                    rewards=rewards,
                    reward_function_name=tracker_name,
                    problem_ids=problem_ids,
                    num_generations=num_gens,
                )

    def _record_reward_events(self, prompts, completions, reward_kwargs, rewards_per_func, reward_func_names):
        """Write one event per completion + stage for step summary."""
        try:
            acc_idx = reward_func_names.index("accuracy_reward")
            fmt_idx = reward_func_names.index("format_reward_func")
        except ValueError:
            acc_idx, fmt_idx = None, None

        G = max(1, int(getattr(self.args, "num_generations", 1)))
        step, ts = int(self.state.global_step), datetime.now().isoformat()
        targets = reward_kwargs.get("target", [])
        problem_ids = reward_kwargs.get("problem_id", [])
        total_completions = len(completions)

        def _phash(txt: str) -> str:
            return hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()[:12]

        for i, (p, c) in enumerate(zip(prompts, completions)):
            pid = _select_for_index(problem_ids, i, G, total_completions)
            target = _select_for_index(targets, i, G, total_completions)

            acc = float(rewards_per_func[i, acc_idx]) if acc_idx is not None else float("nan")
            fmt = float(rewards_per_func[i, fmt_idx]) if fmt_idx is not None else float("nan")
            pred_ans = _extract_boxed_answer(c)

            event = {
                "ts": ts, "step": step, "problem_id": int(pid),
                "sample_index": i // G, "gen_index": i % G,
                "accuracy_reward": acc, "format_reward": fmt,
                "correct": bool(acc >= 0.999), "target": target,
                "pred_answer": pred_ans, "prompt_hash": _phash(p),
            }
            self.reward_logger.log_event(event)

    def log_step(self, logs):
        """Add periodic reward statistics logging and saving."""
        super().log_step(logs)

        if self.accelerator.is_main_process:
            # Flush the previous step for the modern logger
            try:
                if self.reward_logger:
                    prev_step = int(self.state.global_step) - 1
                    if prev_step >= 0:
                        self.reward_logger.flush_step(prev_step)
                        eval_steps = getattr(self.args, "eval_steps", None)
                        is_eval_step = bool(eval_steps and prev_step % eval_steps == 0)
                        if is_eval_step:
                            # Console-only summary (W&B already handled in _finalize_and_log_eval_passk)
                            pass_at_k_metrics = self._calculate_pass_at_k_from_step_summary(prev_step)
                            if pass_at_k_metrics:
                                logger.info(f"Step {prev_step} EVAL pass@k: {pass_at_k_metrics}")
            except Exception as e:
                self.accelerator.print(f"RewardLogger flush_step failed: {e}")

            # Log & snapshot legacy stats
            if self.legacy_tracker:
                self.legacy_tracker.log_recent_stats_for_step(self.state.global_step)
                self.legacy_tracker.save_statistics(step=self.state.global_step)

    def _calculate_pass_at_k_from_step_summary(self, step):
        """Calculate pass@k from reward logger's step summary (for console)."""
        if not self.accelerator.is_main_process or not self.reward_logger:
            return {}
        try:
            step_file = self.reward_logger.step_dir / f"step_{step:06d}.json"
            if not step_file.exists():
                return {}
            with open(step_file, 'r') as f:
                step_data = json.load(f)
            summaries = step_data.get("summaries", {})
            if not summaries:
                return {}
            num_generations = getattr(self.args, "num_generations", 1)
            k_values = self._k_grid(num_generations)
            pass_at_k_metrics = {}
            for k in k_values:
                pass_counts = []
                for problem_summary in summaries.values():
                    if f"pass@{k}" in problem_summary:
                        pass_counts.append(problem_summary[f"pass@{k}"])
                if pass_counts:
                    pass_at_k_metrics[f"pass@{k}"] = sum(pass_counts) / len(pass_counts)
            return pass_at_k_metrics
        except Exception as e:
            self.accelerator.print(f"Pass@k calculation from step summary failed: {e}")
            return {}

    def log_final_pass_at_k_statistics(self):
        """Compute final pass@k statistics across steps (console only)."""
        if not self.accelerator.is_main_process or not self.reward_logger:
            return
        try:
            step_files = list(self.reward_logger.step_dir.glob("step_*.json"))
            if not step_files:
                return
            num_generations = getattr(self.args, "num_generations", 1)
            k_values = self._k_grid(num_generations)
            all_pass_at_k = {k: [] for k in k_values}
            for step_file in step_files:
                try:
                    with open(step_file, 'r') as f:
                        step_data = json.load(f)
                    summaries = step_data.get("summaries", {})
                    if not summaries:
                        continue
                    for k in k_values:
                        pass_counts = []
                        for problem_summary in summaries.values():
                            if f"pass@{k}" in problem_summary:
                                pass_counts.append(problem_summary[f"pass@{k}"])
                        if pass_counts:
                            all_pass_at_k[k].append(sum(pass_counts) / len(pass_counts))
                except Exception as e:
                    self.accelerator.print(f"Error reading step file {step_file}: {e}")
                    continue
            final_metrics = {f"final_pass@{k}": (sum(v)/len(v)) for k, v in all_pass_at_k.items() if v}
            if final_metrics:
                logger.info(f"Final pass@k statistics: {final_metrics}")
        except Exception as e:
            self.accelerator.print(f"Final pass@k statistics calculation failed: {e}")

    # Public helper to make sure pending eval aggregates are flushed at the end
    def finalize_pending_eval_passk(self):
        if self._eval_accumulating:
            self._finalize_and_log_eval_passk()

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
            trainer.log_final_pass_at_k_statistics()  # Console-only final pass@k statistics
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
