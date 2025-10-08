"""
GRPO Training for Mathematical Reasoning (with per-step reward logging and cipher experiment)

This script trains language models using GRPO (Group Relative Policy Optimization)
on mathematical reasoning tasks. The training teaches models to solve math problems
with step-by-step reasoning in a structured format:

- <think>step-by-step reasoning</think>
- \\boxed{final answer}

Cipher Experiment:
- Input prompts (questions and instructions) are ciphered using a permutation mapping
- Original alphabet: abcdefghijklmnopqrstuvwxyz -> udaihveyrcnxobslwpkfgtzjmq
- Model receives ciphered text and generates ciphered responses
- Model outputs are deciphered before reward evaluation

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
from collections import defaultdict

# Setup paths for imports
import sys
from pathlib import Path

# Add TRL root to Python path to access both trl and reward_funcs
script_dir = Path(__file__).parent
trl_root = script_dir.parent.parent  # examples/scripts -> examples -> trl
sys.path.insert(0, str(trl_root))

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
from trl.trainer import IndexedGRPOTrainer
from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig
import wandb

# Import reward functions from the reward_funcs module
from reward_funcs import get_reward_func

# Import system prompts
from prompts import THINK_SYSTEM_PROMPT

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
    
    # LLM Judge configuration for accuracy reward
    use_llm_judge: bool = field(default=False, metadata={"help": "Whether to use LLM judge as fallback for accuracy evaluation"})
    llm_judge_model_name: str = field(default="gpt-5-mini", metadata={"help": "Model name for LLM judge evaluation"})
    llm_judge_api_key_name: str = field(default="OPENAI_API_KEY", metadata={"help": "Environment variable name for LLM judge API key"})
    llm_judge_base_url: str = field(default=None, metadata={"help": "Base URL for LLM judge API endpoint"})
    llm_judge_temperature: float = field(default=0.0, metadata={"help": "Temperature for LLM judge generation"})



########################
# Cipher/Decipher Utilities
########################

# Cipher permutation mapping
ORIGINAL_LOWER = "abcdefghijklmnopqrstuvwxyz"
CIPHER_LOWER = "udaihveyrcnxobslwpkfgtzjmq"
ORIGINAL_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CIPHER_UPPER = "UDAIHVEYRCNXOBSLWPKFGTZJMQ"

# Create translation tables
CIPHER_TABLE = str.maketrans(ORIGINAL_LOWER + ORIGINAL_UPPER, CIPHER_LOWER + CIPHER_UPPER)
DECIPHER_TABLE = str.maketrans(CIPHER_LOWER + CIPHER_UPPER, ORIGINAL_LOWER + ORIGINAL_UPPER)

def cipher_text(text: str) -> str:
    """Apply cipher permutation to text."""
    return text.translate(CIPHER_TABLE)

def decipher_text(text: str) -> str:
    """Reverse cipher permutation to get original text."""
    return text.translate(DECIPHER_TABLE)

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
        project_name = getattr(training_args, "project_name", "grpo-math-training")

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project=project_name,
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
        """Generate prompt with step-by-step thinking format. Applies cipher to question."""
        try:
            # Cipher the question
            ciphered_question = cipher_text(question)
            
            # Cipher the system prompt
            ciphered_system_prompt = cipher_text(THINK_SYSTEM_PROMPT)
            
            # Cipher the instruction text
            instruction = "Think step-by-step inside <think>...</think> tags, then give your final answer inside \\boxed{{}}."
            ciphered_instruction = cipher_text(instruction)
            
            conversation = [
                {
                    "role": "system",
                    "content": ciphered_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"{ciphered_question}\n\n{ciphered_instruction}",
                },
#                {"role": "assistant", "content": "<think>"},
            ]

            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=False, add_generation_prompt=True)
            return {"prompt": prompt, "target": answer}
        except Exception as e:
            logger.error(f"Error generating math prompt: {e}")
            # Fallback to simple format
            ciphered_solve = cipher_text("Solve:")
            ciphered_question = cipher_text(question)
            ciphered_instruction = cipher_text("Think step by step.")
            ciphered_think = cipher_text("<think>")
            return {"prompt": f"{ciphered_solve} {ciphered_question}\n{ciphered_instruction}\n{ciphered_think}", "target": answer}

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

    # Initialize reward functions
    format_reward_func = get_reward_func("format")
    
    accuracy_config = {
        'use_llm_judge': script_args.use_llm_judge,
        'llm_judge_model_name': script_args.llm_judge_model_name,
        'llm_judge_api_key_name': script_args.llm_judge_api_key_name,
        'llm_judge_base_url': script_args.llm_judge_base_url,
        'llm_judge_temperature': script_args.llm_judge_temperature
    }
    accuracy_reward_func = get_reward_func("accuracy", accuracy_config)
    
    if script_args.use_llm_judge:
        logger.info("Using accuracy_reward_func with LLM judge fallback")
    else:
        logger.info("Using accuracy_reward_func with math_verify only")
    
    # Create wrapper reward functions that decipher model output before evaluation
    def deciphering_reward_wrapper(reward_func):
        """Wrapper that deciphers model output before passing to reward function."""
        def wrapped_reward(prompt, completion, target, **kwargs):
            # Decipher the completion before evaluation
            deciphered_completion = decipher_text(completion)
            # Call original reward function with deciphered text
            return reward_func(prompt, deciphered_completion, target, **kwargs)
        return wrapped_reward
    
    # Wrap reward functions with deciphering
    format_reward_func_deciphered = deciphering_reward_wrapper(format_reward_func)
    accuracy_reward_func_deciphered = deciphering_reward_wrapper(accuracy_reward_func)
    
    reward_functions = [format_reward_func_deciphered, accuracy_reward_func_deciphered]
    
    logger.info("Cipher experiment enabled: prompts will be ciphered, outputs will be deciphered for reward evaluation")
    trainer = IndexedGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        tokenizer=tokenizer,
    )

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

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")

        # Ensure model is in the same dtype as training before uploading
        if hasattr(training_args, 'torch_dtype') and training_args.torch_dtype:
            target_dtype = training_args.torch_dtype
            if hasattr(trainer.model, 'dtype') and trainer.model.dtype != target_dtype:
                logger.info(f"Converting model from {trainer.model.dtype} to {target_dtype} for hub upload")
                trainer.model.to(target_dtype)

        trainer.push_to_hub(commit_message=f"GRPO training checkpoint - Step {trainer.state.global_step}")

    # Flush any pending eval pass@k (e.g., if training ended immediately after eval)
    trainer.finalize_pending_eval_passk()

    # Finalize all logging
    if trainer.accelerator.is_main_process:
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
