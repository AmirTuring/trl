"""
IndexedGRPOTrainer - Enhanced GRPO Trainer with Pass@k Metrics and Detailed Logging

This module provides an enhanced version of GRPOTrainer that adds:
- Per-step reward logging with pass@k metrics
- Detailed reward statistics tracking
- Proper evaluation mode handling
- WandB logging integration with HuggingFace-style cadence
- Step boundary detection and metric accumulation

Features:
- Pass@k metrics with **Hugging Face-style cadence**:
  - **Train:** averaged over steps since the last log (windowed by `logging_steps`).
  - **Eval:** aggregated over the entire eval loop at each `eval_steps` point (single log per eval run).
- Per-completion event logging (JSONL) updated **every step**
- Per-step summaries with mean/std per prompt and pass@k
- Separate folder per run: <output_dir>/reward_logs/<run_name>_<timestamp>/
- Proper WandB metric naming: `train/pass_at_k/{k}` and `eval/pass_at_k/{k}`
- **Only rank 0 reads/prints/logs** - no cross-rank reductions needed
- **Pass@k computed from accumulated data** within steps
- **Step boundary detection** - flushes step summaries when global step advances
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict
import torch

# Check for wandb availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .grpo_trainer import GRPOTrainer

logger = logging.getLogger(__name__)


class IndexedGRPOTrainer(GRPOTrainer):
    """
    Enhanced GRPO Trainer with advanced pass@k metrics and detailed reward logging.
    
    This trainer extends the base GRPOTrainer to add comprehensive metrics tracking,
    including pass@k calculations using the correct combinatorial formula, detailed
    reward statistics per problem, and proper evaluation/training mode handling.
    """
    
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

        # Track evaluation mode
        self._is_evaluation_mode = False

        # Accumulate data for pass@k calculation across logging steps (HF-style)
        self._step_correct_answers = []
        self._step_problem_ids = []
        self._last_logged_step = -1

        # Accumulate pass@k metrics across logging window
        self._logging_window_passk_data = []  # List of pass@k dicts from each step
        self._logging_window_start_step = 0

        # Accumulate detailed reward statistics per problem
        self._step_reward_details = defaultdict(dict)

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
            acc_idx = reward_func_names.index("accuracy_reward_func")

            # Get accuracy rewards and determine correct answers
            accuracy_rewards = rewards_per_func[:, acc_idx].cpu().tolist()
            correct_answers = [1.0 if acc >= 0.999 else 0.0 for acc in accuracy_rewards]

            # Get problem IDs
            problem_ids = reward_kwargs.get("problem_id", [])

            logger.debug(f"_ACCUMULATE_DATA: Processing {len(correct_answers)} answers, {len(problem_ids)} problem IDs, "
                        f"accuracy rewards mean: {sum(accuracy_rewards)/len(accuracy_rewards):.4f}")

            # Accumulate data for pass@k calculation
            self._step_correct_answers.extend(correct_answers)
            self._step_problem_ids.extend(problem_ids)

            # Accumulate detailed reward statistics per problem
            try:
                # Get rewards for each reward function by name
                reward_arrays = {}
                for func_idx, func_name in enumerate(self.reward_func_names):
                    reward_arrays[func_name] = rewards_per_func[:, func_idx].cpu().tolist()

                # Accumulate rewards per problem for each function
                for i, problem_id in enumerate(problem_ids):
                    pid = int(problem_id)
                    if pid not in self._step_reward_details:
                        self._step_reward_details[pid] = {func_name: [] for func_name in self.reward_func_names}

                    for func_name in self.reward_func_names:
                        self._step_reward_details[pid][func_name].append(reward_arrays[func_name][i])

                logger.debug(f"_ACCUMULATE_DATA: Accumulated rewards for {len(set(problem_ids))} unique problems")

            except Exception as e:
                logger.warning(f"Failed to accumulate detailed reward statistics: {e}")

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
                    if WANDB_AVAILABLE and wandb.run:
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

            # Save detailed reward statistics
            try:
                if self._step_reward_details:
                    self._save_detailed_reward_statistics(self.state.global_step, is_evaluation=False)
            except Exception as e:
                self.accelerator.print(f"Detailed reward statistics save failed: {e}")

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
        self._step_reward_details.clear()

        # Set evaluation mode for proper metric logging
        self._is_evaluation_mode = True

        out = super().evaluate(*args, **kwargs)

        if self.accelerator.is_main_process:
            # Calculate pass@k from accumulated step data
            if self._step_correct_answers and self._step_problem_ids:
                final_metrics = self._calculate_pass_at_k_once_per_step()

                if final_metrics and WANDB_AVAILABLE and wandb.run:
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

            # Save detailed reward statistics for evaluation
            try:
                if self._step_reward_details:
                    self._save_detailed_reward_statistics(self.state.global_step, is_evaluation=True)
            except Exception as e:
                logger.warning(f"Failed to save evaluation reward statistics: {e}")

        # Reset window after logging and reset evaluation mode
        self._logging_window_passk_data.clear()
        self._is_evaluation_mode = False

        # Clear step data after evaluation to prepare for next training step
        logger.info(f"EVALUATE METHOD: Clearing step data after evaluation - had {len(self._step_correct_answers)} answers")
        self._step_correct_answers.clear()
        self._step_problem_ids.clear()
        self._step_reward_details.clear()

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
                    if WANDB_AVAILABLE and wandb.run:
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

    def _save_detailed_reward_statistics(self, step: int, is_evaluation: bool = False):
        """Save detailed reward statistics for the current step."""
        try:
            if not self.accelerator.is_main_process or not self._step_reward_details:
                return

            # Create separate directories for train and eval
            mode_dir = "eval" if is_evaluation else "train"
            filename = f"reward_details_step_{step:06d}.json"
            filepath = os.path.join(self.args.output_dir, "reward_logs", mode_dir, filename)

            # Prepare statistics for each problem
            reward_stats = {}
            for problem_id, rewards in self._step_reward_details.items():
                import numpy as np
                problem_stats = {
                    'num_generations': len(next(iter(rewards.values()))) if rewards else 0
                }

                # Add reward arrays and statistics for each function
                for func_name, reward_values in rewards.items():
                    problem_stats[f'{func_name}_rewards'] = reward_values
                    problem_stats[f'{func_name}_mean'] = float(np.mean(reward_values)) if reward_values else 0.0
                    problem_stats[f'{func_name}_std'] = float(np.std(reward_values)) if reward_values else 0.0

                reward_stats[str(problem_id)] = problem_stats

            # Save to file
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({
                    'step': step,
                    'timestamp': datetime.now().isoformat(),
                    'is_evaluation': is_evaluation,
                    'num_problems': len(reward_stats),
                    'reward_statistics': reward_stats
                }, f, indent=2)

            logger.info(f"Saved detailed reward statistics for {len(reward_stats)} problems at step {step} to {filepath}")

            # Clear accumulated data after saving (only during training, keep during eval)
            if not is_evaluation:
                self._step_reward_details.clear()
                logger.debug(f"Cleared reward details after training step {step}")
            else:
                logger.debug(f"Keeping reward details for evaluation at step {step}")

        except Exception as e:
            self.accelerator.print(f"Failed to save detailed reward statistics: {e}")

    def _k_grid(self, num_generations: int):
        """Generate k values as powers of 2 until num_generations / 2."""
        k_values = [1]
        current_k = 2
        max_k = max(1, num_generations // 2)  # Powers of 2 until num_generations / 2
        while current_k <= max_k:
            k_values.append(current_k)
            current_k *= 2
        return sorted(list(set(k_values)))
