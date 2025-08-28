"""
Reward Logging Utilities for GRPO Mathematical Reasoning Training

This module provides classes for managing and logging reward data during training.
It includes:
- RewardLogger: For streaming per-completion events and step summaries.
- LegacyRewardTracker: For maintaining and saving aggregate reward statistics
  in a manner consistent with previous script versions.
"""

import os
import re
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from math import comb
from collections import defaultdict
from typing import List, Dict, Any
from dataclasses import asdict # <-- ADDED IMPORT

import numpy as np

# Setup logger for this module
logger = logging.getLogger(__name__)


def _sanitize_name(s: str) -> str:
    """Remove characters from a string that are not suitable for filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_")


def _select_for_index(arr, i, G, total_len):
    """Robustly select an item from an array for a given completion index."""
    if isinstance(arr, (list, tuple)):
        if len(arr) == total_len:
            return arr[i]
        if len(arr) * G == total_len:
            return arr[i // max(1, G)]
        if len(arr) == 1:
            return arr[0]
        # Fallback
        return arr[min(i, len(arr) - 1)]
    return arr


class RewardLogger:
    """
    Streams per-completion events and writes per-step summaries with pass@k.
    Files live under: <output_dir>/reward_logs/<run_name>_<timestamp>/
    Writes only on main process.
    """
    def __init__(self, base_dir: str, run_name: str, is_main: bool, num_generations: int, k_defaults=(1, 5)):
        self.is_main = is_main
        self.num_generations = int(num_generations)
        self.k_defaults = tuple(sorted(set([k for k in k_defaults if k >= 1])))

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(base_dir) / f"{_sanitize_name(run_name)}_{ts}"
        self.step_dir = self.run_dir / "step_summaries"
        if self.is_main:
            self.step_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.meta_path = self.run_dir / "meta.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.overall_path = self.run_dir / "overall.json"

        # In-memory buffer: step -> problem_id -> list of event dicts
        self._buffer = defaultdict(lambda: defaultdict(list))
        # Running aggregates for overall.json (across flushed steps)
        self._overall = {
            "total_events": 0,
            "total_prompts": 0,
            "steps_flushed": 0,
            "k_values": None,  # filled at first flush
            "by_step": {},     # step -> {"prompts": N, "events": M}
        }

    def write_meta(self, model_args, training_args, dataset_args):
        if not self.is_main:
            return
            
        # --- MODIFICATION START ---
        # Convert DatasetConfig objects to dictionaries to make them JSON serializable
        datasets_info = None
        raw_datasets_configs = getattr(dataset_args, "datasets", None)
        if raw_datasets_configs:
            datasets_info = [asdict(config) for config in raw_datasets_configs]
        # --- MODIFICATION END ---

        meta = {
            "created_at": datetime.now().isoformat(),
            "run_name": getattr(training_args, "run_name", "run"),
            "output_dir": training_args.output_dir,
            "num_generations": self.num_generations,
            "k_defaults": list(self.k_defaults),
            "accuracy_correct_threshold": 0.999,
            "model": {
                "name_or_path": model_args.model_name_or_path,
                "revision": model_args.model_revision,
            },
            "training": {
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "grad_accum": training_args.gradient_accumulation_steps,
                "max_steps": getattr(training_args, "max_steps", None),
                "num_train_epochs": getattr(training_args, "num_train_epochs", None),
            },
            "datasets": datasets_info, # <-- Use the converted list of dicts
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def log_event(self, event: Dict[str, Any]):
        """
        event must include keys: step, problem_id, gen_index, accuracy_reward, format_reward
        Optional: prompt_hash, target, pred_answer, sample_index
        """
        if not self.is_main:
            return

        # Append to JSONL
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Stage into in-memory buffer for step summary writing
        step = int(event["step"])
        pid = int(event["problem_id"])
        self._buffer[step][pid].append(event)

    @staticmethod
    def _pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
        # Standard "at least one success in k draws without replacement" estimator
        if num_correct <= 0 or k <= 0 or num_samples <= 0:
            return 0.0
        k = min(k, num_samples)
        if num_correct >= num_samples:
            return 1.0
        # 1 - C(n-c, k) / C(n, k)
        num = comb(num_samples - num_correct, k)
        den = comb(num_samples, k)
        return 1.0 - float(num) / float(den)

    def _summarize_problem(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        acc = [float(e["accuracy_reward"]) for e in events]
        fmt = [float(e["format_reward"]) for e in events]
        correct_bools = [bool(e.get("correct", False)) for e in events]
        c = sum(1 for x in correct_bools if x)
        n = len(events)
        # Determine k values: 1, defaults, configured G, observed n
        k_values = sorted(set([1, *self.k_defaults, self.num_generations, n]))
        k_values = [k for k in k_values if k <= n]  # clip to observed samples in this step

        pass_at = {f"pass@{k}": self._pass_at_k(c, n, k) for k in k_values}

        def _stats(xs):
            if not xs:
                return {"mean": 0.0, "std": 0.0}
            return {"mean": float(np.mean(xs)), "std": float(np.std(xs))}

        example = events[0]
        return {
            "problem_id": int(example["problem_id"]),
            "num_gens": n,
            "num_correct": int(c),
            "acc": _stats(acc),
            "fmt": _stats(fmt),
            **pass_at,
            "target": example.get("target"),
            "prompt_hash": example.get("prompt_hash"),
        }

    def flush_step(self, step_to_flush: int):
        """Write step_summaries/step_<step>.json and update overall.json."""
        if not self.is_main or step_to_flush < 0:
            return
        if step_to_flush not in self._buffer:
            return

        problems = self._buffer.pop(step_to_flush)
        summaries = {}
        for pid, evts in problems.items():
            summaries[str(pid)] = self._summarize_problem(evts)

        # Write step summary
        out_path = self.step_dir / f"step_{step_to_flush:06d}.json"
        out_path.write_text(json.dumps({
            "step": int(step_to_flush),
            "created_at": datetime.now().isoformat(),
            "num_prompts": len(summaries),
            "summaries": summaries,
        }, indent=2))

        # Update overall.json (running aggregate)
        self._overall["steps_flushed"] += 1
        self._overall["total_prompts"] += len(summaries)
        step_events = sum(len(problems[int(pid)]) for pid in summaries.keys())
        self._overall["total_events"] += step_events
        self._overall["k_values"] = self._overall["k_values"] or list(
            sorted({1, *self.k_defaults, self.num_generations})
        )
        self._overall["by_step"][str(step_to_flush)] = {
            "prompts": len(summaries),
            "events": step_events,
        }
        self.overall_path.write_text(json.dumps(self._overall, indent=2))

    def finalize(self):
        """Finalize logging. Currently a no-op as files are written continuously."""
        pass


class LegacyRewardTracker:
    """
    Manages in-memory storage and JSON snapshotting of reward statistics.
    This class encapsulates the global REWARD_STORAGE and its related functions
    from the original script. It only performs actions on the main process.
    """
    def __init__(self, is_main: bool):
        self.is_main = is_main
        if not self.is_main:
            return

        self.storage = {
            "format_reward": defaultdict(list),
            "accuracy_reward": defaultdict(list),
        }
        self.output_dir = "reward_statistics"
        os.makedirs(self.output_dir, exist_ok=True)

    def update_batch(
        self,
        rewards: List[float],
        reward_function_name: str,
        problem_ids: List[Any],
        num_generations: int,
    ):
        """Update the tracker with a batch of rewards for a specific function."""
        if not self.is_main or reward_function_name not in self.storage:
            return

        G = max(1, int(num_generations))
        total_rewards = len(rewards)

        try:
            if not problem_ids:
                problem_ids = list(range(total_rewards))
            for i, r in enumerate(rewards):
                pid = _select_for_index(problem_ids, i, G, total_rewards)
                self.storage[reward_function_name][int(pid)].append(float(r))
        except Exception as e:
            logger.warning(f"Failed to update legacy reward tracker for {reward_function_name}: {e}")

    def get_problem_stats(self, problem_id: Any, reward_function_name: str) -> Dict[str, Any]:
        """Get reward statistics for a specific problem and reward function."""
        if not self.is_main:
            return None

        rewards = self.storage.get(reward_function_name, {}).get(problem_id, [])
        if not rewards:
            return None

        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "count": len(rewards),
        }

    def save_statistics(self, step: int = None):
        """Save current reward statistics to JSON files."""
        if not self.is_main:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_suffix = f"_step_{step}" if step is not None else ""

        for func_name, storage in self.storage.items():
            stats = {str(pid): self.get_problem_stats(pid, func_name) for pid in storage if storage[pid]}
            filename = os.path.join(self.output_dir, f"{func_name}_stats{step_suffix}_{timestamp}.json")
            with open(filename, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved {func_name} statistics for {len(stats)} problems to {filename}")

        step_filename = os.path.join(self.output_dir, f"step_{step}_rewards.json")
        step_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "reward_storage": {k: {str(pid): v for pid, v in d.items()} for k, d in self.storage.items()},
        }
        with open(step_filename, "w") as f:
            json.dump(step_data, f, indent=2)
        logger.info(f"Saved step {step} raw reward data to {step_filename}")

    def print_summary(self):
        """Print a summary of reward statistics to the console."""
        if not self.is_main:
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("REWARD STATISTICS SUMMARY")
        logger.info("=" * 60)

        for func_name, storage in self.storage.items():
            if not storage:
                logger.info(f"\n{func_name.upper()}: No data recorded")
                continue

            logger.info(f"\n{func_name.upper()} REWARDS:")
            all_rewards = [r for rewards_list in storage.values() for r in rewards_list]

            if all_rewards:
                problem_stats = [(pid, self.get_problem_stats(pid, func_name)) for pid in storage]
                problem_stats = [(pid, stats) for pid, stats in problem_stats if stats]

                problem_means = [stats['mean'] for _, stats in problem_stats]

                logger.info(f"  Problems with recorded rewards: {len(problem_stats)}")
                logger.info(f"  Total reward evaluations: {len(all_rewards)}")
                logger.info(f"  Overall mean: {np.mean(all_rewards):.3f} (std: {np.std(all_rewards):.3f})")
                if problem_means:
                    logger.info(f"  Problem-level mean: {np.mean(problem_means):.3f} (std: {np.std(problem_means):.3f})")

                if problem_stats:
                    problem_stats.sort(key=lambda x: x[1]["mean"], reverse=True)
                    logger.info("\n  Top 3 performing problems:")
                    for pid, stats in problem_stats[:3]:
                        logger.info(f"    Problem {pid}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} evals)")

                    logger.info("\n  Bottom 3 performing problems:")
                    for pid, stats in problem_stats[-3:]:
                        logger.info(f"    Problem {pid}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} evals)")
        logger.info("\n" + "=" * 60)

    def log_recent_stats_for_step(self, step: int):
        """Log a brief summary of recent activity for the current step."""
        if not self.is_main:
            return

        logger.info(f"\n--- Reward Statistics Snapshot at Step {step} ---")
        for func_name, storage in self.storage.items():
            if storage:
                recent_problems = list(storage.keys())[-5:]
                logger.info(f"\n{func_name} rewards (last 5 problems):")
                for pid in recent_problems:
                    stats = self.get_problem_stats(pid, func_name)
                    if stats and stats["count"] > 0:
                        logger.info(f"  Problem {pid}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} evals)")