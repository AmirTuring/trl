"""
GRPO Training for Mathematical Reasoning

This script trains language models using GRPO (Group Relative Policy Optimization) 
on mathematical reasoning tasks. The training teaches models to solve math problems 
with step-by-step reasoning in a structured format:

- <think>step-by-step reasoning</think>
- <answer>final answer</answer>

Reward functions:
1. Format reward: Ensures proper <think></think><answer></answer> structure
2. Accuracy reward: Uses math_verify to evaluate mathematical correctness

Usage:
    python grpo_math.py --config examples/cli_configs/grpo_math_config.yaml
"""

import logging
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import json

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser, DatasetMixtureConfig, get_dataset, ScriptArguments as TrlScriptArguments
from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig
import wandb
import torch
import numpy as np

# Login to WandB using environment variable
wandb_token = os.getenv("WANDB_API_KEY")
if wandb_token:
    wandb.login(key=wandb_token)

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

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
# Global reward storage for tracking per-problem rewards
########################

# Dictionary to store rewards per dataset item for each reward function
REWARD_STORAGE = {
    'format_reward': {},  # dataset_index -> list of rewards
    'accuracy_reward': {}  # dataset_index -> list of rewards
}

def get_problem_reward_stats(dataset_index, reward_function_name):
    """Get reward statistics for a specific problem and reward function."""
    if dataset_index not in REWARD_STORAGE[reward_function_name]:
        return None
    
    rewards = REWARD_STORAGE[reward_function_name][dataset_index]
    if not rewards:
        return None
    
    import numpy as np
    return {
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
        'count': len(rewards),
        'rewards': rewards
    }

def save_reward_statistics():
    """Save current reward statistics to JSON files."""
    os.makedirs("reward_statistics", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for func_name, storage in REWARD_STORAGE.items():
        stats = {}
        for dataset_idx, rewards in storage.items():
            if rewards:  # Only include problems with recorded rewards
                stats[str(dataset_idx)] = get_problem_reward_stats(dataset_idx, func_name)
        
        filename = f"reward_statistics/{func_name}_stats_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved {func_name} statistics for {len(stats)} problems to {filename}")

def print_reward_summary():
    """Print summary of reward statistics per problem."""
    logger.info("\n" + "="*60)
    logger.info("REWARD STATISTICS SUMMARY")
    logger.info("="*60)
    
    for func_name, storage in REWARD_STORAGE.items():
        if not storage:
            logger.info(f"\n{func_name.upper()}: No data recorded")
            continue
            
        logger.info(f"\n{func_name.upper()} REWARDS:")
        logger.info(f"  Problems with recorded rewards: {len(storage)}")
        
        # Calculate overall statistics
        all_rewards = []
        problem_means = []
        for dataset_idx, rewards in storage.items():
            if rewards:
                all_rewards.extend(rewards)
                problem_means.append(sum(rewards) / len(rewards))
        
        if all_rewards:
            import numpy as np
            logger.info(f"  Total reward evaluations: {len(all_rewards)}")
            logger.info(f"  Overall mean: {np.mean(all_rewards):.3f}")
            logger.info(f"  Overall std: {np.std(all_rewards):.3f}")
            logger.info(f"  Problem-level mean: {np.mean(problem_means):.3f}")
            logger.info(f"  Problem-level std: {np.std(problem_means):.3f}")
            
            # Show best and worst performing problems
            problem_stats = [(idx, get_problem_reward_stats(idx, func_name)) 
                           for idx in storage.keys()]
            problem_stats = [(idx, stats) for idx, stats in problem_stats if stats is not None]
            
            if problem_stats:
                # Sort by mean reward
                problem_stats.sort(key=lambda x: x[1]['mean'], reverse=True)
                
                logger.info(f"\n  Top 3 performing problems:")
                for i, (idx, stats) in enumerate(problem_stats[:3]):
                    logger.info(f"    Problem {idx}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} evals)")
                
                logger.info(f"\n  Bottom 3 performing problems:")
                for i, (idx, stats) in enumerate(problem_stats[-3:]):
                    logger.info(f"    Problem {idx}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} evals)")
    
    logger.info("\n" + "="*60)

########################
# Helper functions
########################

def format_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on strict format:
      <think> ...nonempty reasoning... </think> followed by exactly one \\boxed{...}
    
    Rewarding scheme:
      -1.0 : invalid / reward hacking attempt
       0.0 : partially correct but missing constraints
       1.0 : valid format
    """
    rewards = []
    
    # Get dataset indices for tracking rewards per problem
    dataset_indices = kwargs.get('dataset_indices', [])

    for completion in completions:
        try:
            # Ensure consistency with prompt (prepend <think>)
            completion = "<think>" + completion

            # Occasionally log completions for inspection
            if random.random() < 0.01:
                os.makedirs("completion_samples", exist_ok=True)
                with open("completion_samples/format_completion_samples.txt", "a") as f:
                    f.write("\n\n==============\n")
                    f.write(completion)

            # Match a single <think>...</think> block at the start
            think_pattern = r"^<think>(.*?)</think>"
            think_match = re.search(think_pattern, completion, re.DOTALL)

            if not think_match:
                rewards.append(-1.0)  # No valid reasoning block
                continue

            think_content = think_match.group(1).strip()
            after_think = completion[think_match.end():]

            # Reject if <think> is empty or trivially short
            if len(think_content) < 10:
                rewards.append(-1.0)
                continue

            # Reject if another <think> tag appears later (reward hacking)
            if "<think>" in after_think:
                rewards.append(-1.0)
                continue

            # Count \boxed occurrences
            boxed_matches = re.findall(r"\\boxed\{.*?\}", completion)

            if len(boxed_matches) != 1:
                rewards.append(-1.0)  # Must be exactly one answer
                continue

            # Ensure the \boxed appears *after* reasoning
            if "\\boxed{" not in after_think:
                rewards.append(-1.0)
                continue

            # Penalize flooding after </think>
            if len(after_think.strip()) > 4 * len(think_content):
                rewards.append(0.0)  # suspiciously long tail
                continue

            # Passed all checks
            rewards.append(1.0)

        except Exception:
            rewards.append(0.0)

    # Store rewards per dataset index
    if dataset_indices and len(dataset_indices) == len(rewards):
        for idx, reward in zip(dataset_indices, rewards):
            if idx not in REWARD_STORAGE['format_reward']:
                REWARD_STORAGE['format_reward'][idx] = []
            REWARD_STORAGE['format_reward'][idx].append(reward)
    
    return rewards

def accuracy_reward(completions, target, **kwargs):
    """
    Reward function that evaluates mathematical correctness using math_verify.
    
    Args:
        completions (list[str]): Generated outputs containing \\boxed{} answers
        target (list[str]): Ground truth answers
        
    Returns:
        list[float]: Reward scores (1.0 for correct, 0.0 for incorrect)
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    rewards = []
    
    # Get dataset indices for tracking rewards per problem
    dataset_indices = kwargs.get('dataset_indices', [])
    
    for completion, ground_truth in zip(completions, target):
        try:
            # Extract answer from \boxed{} format
            completion_answer = _extract_boxed_answer(completion)
            
            completion_answer = "\\boxed{" + completion_answer + "}"
            
            # Wrap ground truth in \boxed{} format for verification
            ground_truth_boxed = "\\boxed{" + ground_truth + "}"
            
            # First try exact match (case-insensitive)
            if ground_truth_boxed.strip().lower() == completion_answer.strip().lower():
                reward = 1.0
            else:
                # Use math verification for semantic comparison
                score, _ = verify_func([ground_truth_boxed], [completion_answer])
                reward = float(score)
                
        except Exception:
            reward = 0.0
            
        rewards.append(reward)
    
    # Store rewards per dataset index
    if dataset_indices and len(dataset_indices) == len(rewards):
        for idx, reward in zip(dataset_indices, rewards):
            if idx not in REWARD_STORAGE['accuracy_reward']:
                REWARD_STORAGE['accuracy_reward'][idx] = []
            REWARD_STORAGE['accuracy_reward'][idx].append(reward)
    
    return rewards


def _extract_boxed_answer(completion):
    """Extract answer from \\boxed{} format, handling nested braces."""
    if "\\boxed{" not in completion:
        return completion
    
    start_idx = completion.find("\\boxed{") + 7
    brace_count = 1
    end_idx = start_idx
    
    while end_idx < len(completion) and brace_count > 0:
        if completion[end_idx] == '{':
            brace_count += 1
        elif completion[end_idx] == '}':
            brace_count -= 1
        end_idx += 1
    
    return completion[start_idx:end_idx-1].strip()


class IndexedGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer that passes dataset indices to reward functions.
    
    This allows reward functions to track rewards per dataset item (problem).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store dataset indices mapping
        self._dataset_indices = {}
    
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Override to pass dataset indices to reward functions."""
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state
        
        # Add dataset indices to reward_kwargs
        dataset_indices = []
        for example in inputs:
            if hasattr(example, 'get') and 'dataset_index' in example:
                dataset_indices.append(example['dataset_index'])
            elif hasattr(example, '__getitem__') and 'dataset_index' in example:
                dataset_indices.append(example['dataset_index'])
            else:
                # If no explicit index, try to extract from the example or use a default
                dataset_indices.append(getattr(example, '_index', len(dataset_indices)))
        
        reward_kwargs["dataset_indices"] = dataset_indices

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            # Import required functions locally
            from trl.trainer.utils import profiling_context
            from trl.extras.dataset_formatting import is_conversational, apply_chat_template
            
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, torch.nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        from trl.trainer.utils import gather
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func
    
    def get_train_dataloader(self):
        """Override to add dataset indices to the data."""
        dataloader = super().get_train_dataloader()
        
        # Add index to each sample
        if hasattr(dataloader.dataset, '__iter__'):
            # For iterable datasets, we'll add indices dynamically
            original_dataset = dataloader.dataset
            
            class IndexedDataset:
                def __init__(self, dataset):
                    self.dataset = dataset
                    self._current_index = 0
                
                def __iter__(self):
                    self._current_index = 0
                    for item in self.dataset:
                        item['dataset_index'] = self._current_index
                        self._current_index += 1
                        yield item
                
                def __len__(self):
                    return len(self.dataset)
            
            dataloader.dataset = IndexedDataset(original_dataset)
        else:
            # For map-style datasets
            def add_index(example, idx):
                example['dataset_index'] = idx
                return example
            
            dataloader.dataset = dataloader.dataset.map(add_index, with_indices=True)
        
        return dataloader
    
    def log_step(self, logs):
        """Override to add periodic reward statistics logging."""
        super().log_step(logs)
        
        # Log reward statistics every 50 steps
        if self.state.global_step % 50 == 0:
            logger.info(f"\n--- Reward Statistics at Step {self.state.global_step} ---")
            
            for func_name, storage in REWARD_STORAGE.items():
                if storage:
                    recent_problems = list(storage.keys())[-5:]  # Last 5 problems
                    logger.info(f"\n{func_name} rewards (last 5 problems):")
                    
                    for idx in recent_problems:
                        stats = get_problem_reward_stats(idx, func_name)
                        if stats and stats['count'] > 0:
                            logger.info(f"  Problem {idx}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} evals)")


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig, dataset_args: DatasetMixtureConfig
):
    #########################
    # Initialize WandB logging (only on main process)
    #########################
    is_main_process = training_args.local_rank in [-1, 0]
    
    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to and is_main_process:
        wandb_config = {
            "model": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs,
            "beta": training_args.beta,
            "max_prompt_length": training_args.max_prompt_length,
            "max_completion_length": training_args.max_completion_length,
            "num_generations": training_args.num_generations,
            "total_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size,
            "world_size": training_args.world_size,
        }
        
        # Add run name if specified
        run_name = getattr(training_args, 'run_name', 'grpo-math-training')
        
        wandb.init(
            project="grpo-math-training",
            name=run_name,
            config=wandb_config,
            tags=["grpo", "math", "reasoning"],
            # Ensure WandB works properly with distributed training
            settings=wandb.Settings(start_method="fork")
        )
        logger.info(f"WandB logging initialized on main process (world_size: {training_args.world_size})")
    else:
        # Disable wandb on non-main processes to ensure only main process logs
        os.environ["WANDB_MODE"] = "disabled"
        logger.info(f"WandB disabled on rank {training_args.local_rank}")
    
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
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
                # Create a DatasetMixtureConfig for validation datasets
                from trl.scripts.utils import DatasetConfig
                validation_dataset_configs = []
                for val_dataset in script_args.validation_datasets:
                    validation_dataset_configs.append(DatasetConfig(**val_dataset))
                
                validation_mixture_config = DatasetMixtureConfig(
                    datasets=validation_dataset_configs,
                    streaming=dataset_args.streaming,
                    test_split_size=None
                )
                validation_dataset_dict = get_dataset(validation_mixture_config)
                eval_dataset = validation_dataset_dict.get(script_args.dataset_test_split)
                if eval_dataset is not None:
                    logger.info(f"Validation dataset size: {len(eval_dataset)} (using separate validation datasets)")
                else:
                    available_splits = list(validation_dataset_dict.keys())
                    logger.info(f"No validation dataset found for split '{script_args.dataset_test_split}' in validation datasets. Available splits: {available_splits}")
                    # Try to use the first available split
                    if available_splits:
                        eval_dataset = validation_dataset_dict[available_splits[0]]
                        logger.info(f"Using first available split '{available_splits[0]}' for validation. Size: {len(eval_dataset)}")
            else:
                # Use the same dataset with different split
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
                    "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."
                },
                {
                    "role": "user",
                    "content": f"{question}\n\nThink step-by-step inside <think>...</think> tags, then give your final answer inside \\boxed{{}}."
                },
                {
                    "role": "assistant",
                    "content": "<think>"
                }
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
    
    # Convert dataset to prompt format
    def extract_and_format_train(row):
        """Extract question and answer using training field mapping."""
        question = row[train_question_field]
        answer = row[train_answer_field]
        
        prompt_data = generate_math_prompt(question, answer)
        
        return prompt_data  # Return only {"prompt": prompt, "target": answer}
    
    def extract_and_format_eval(row):
        """Extract question and answer using validation field mapping."""
        question = row[eval_question_field]
        answer = row[eval_answer_field]
        
        prompt_data = generate_math_prompt(question, answer)
        
        return prompt_data  # Return only {"prompt": prompt, "target": answer}
    
    logger.info("Converting dataset to prompt format...")
    train_dataset = train_dataset.map(extract_and_format_train, desc="Processing train dataset")
    
    # Process validation dataset if available
    if eval_dataset is not None:
        logger.info("Converting validation dataset to prompt format...")
        eval_dataset = eval_dataset.map(extract_and_format_eval, desc="Processing validation dataset")
    
    # Verify the dataset structure
    logger.info("Verifying dataset structure...")
    sample = train_dataset[0]
    required_fields = ["prompt", "target"]
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        logger.error(f"Missing required fields in dataset: {missing_fields}")
        logger.error(f"Available fields: {list(sample.keys())}")
        raise ValueError(f"Dataset missing required fields: {missing_fields}")
    
    logger.info(f"✓ Dataset validation passed. Sample prompt length: {len(sample['prompt'])}")
    if len(sample['prompt']) > 1000:  # Show truncated version for long prompts
        logger.info(f"Sample prompt (first 200 chars): {sample['prompt'][:200]}...")
    else:
        logger.info(f"Sample prompt: {sample['prompt']}")
    logger.info(f"Sample target: {sample['target']}")
    
    # Set reward functions for math problems
    reward_functions = [format_reward_func, accuracy_reward]

    #########################
    # Instantiate Indexed GRPO trainer
    #########################

    trainer = IndexedGRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=reward_functions,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    if hasattr(training_args, 'max_steps') and training_args.max_steps is not None and training_args.max_steps > 0:
        training_info = f"{training_args.max_steps} steps"
    else:
        training_info = f"{training_args.num_train_epochs} epochs"
    
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_info} ***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    if eval_dataset is not None:
        metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    
    # Log final training metrics to WandB (only on main process)
    # The trainer's built-in logging handles metric aggregation across GPUs
    if wandb.run is not None and trainer.accelerator.is_main_process:
        wandb_final_metrics = {
            "final_train_samples": len(train_dataset),
            "training_completed": True
        }
        if eval_dataset is not None:
            wandb_final_metrics["final_eval_samples"] = len(eval_dataset)
        wandb.log(wandb_final_metrics)

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo"]})
        
        # Save and log reward statistics per problem
        logger.info("*** Saving reward statistics per problem ***")
        save_reward_statistics()
        
        # Print summary statistics
        print_reward_summary()
    
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        
        # Create commit message with current training info
        commit_message = f"GRPO training checkpoint - Step {trainer.state.global_step}"
        
        logger.info(f"Pushing to hub with message: {commit_message}")
        trainer.push_to_hub(commit_message=commit_message)

    # Finish WandB run (only on main process)
    if wandb.run is not None:
        wandb.finish()
        logger.info("WandB logging finished")

    logger.info("*** Training complete! ***")


def main():
    """
    Main entry point for GRPO mathematical reasoning training.
    
    Loads configuration from YAML file and runs GRPO training with:
    - Dataset loading and field mapping (separate for train/validation)
    - Optional validation dataset support
    - Format and accuracy reward functions
    - Step-based training with VLLM inference
    - WandB logging with training and validation metrics
    """
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args, dataset_args)


if __name__ == "__main__":
    main()