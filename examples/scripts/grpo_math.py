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

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser, DatasetMixtureConfig, get_dataset, ScriptArguments as TrlScriptArguments
from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig
import wandb

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

########################
# Helper functions
########################

def format_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on correct format: exactly one <think>...</think> followed by exactly one \\boxed{} answer
    
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers (not used in format checking)
      
    Returns:
        list[float]: Reward scores (1.0 for correct format, 0.0 otherwise)
    """
    rewards = []

    for completion in completions:
        try:
            # Add synthetic <think> as it's already part of the prompt and prefilled
            completion = "<think>" + completion
            
            if random.random() < 0.01:  # 1% chance to write samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "format_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)
            
            # Check for exactly one <think> and one </think>
            think_open_count = completion.count("<think>")
            think_close_count = completion.count("</think>")
            
            if think_open_count != 1 or think_close_count != 1:
                rewards.append(0.0)
                continue
                
            # Extract the think section and find where it ends
            think_match = re.search(r"<think>(.*?)<\/think>", completion, re.DOTALL)
            if not think_match:
                rewards.append(0.0)
                continue
                
            # Extract the part after </think>
            after_think = completion[think_match.end():]
            
            # Count \boxed{} occurrences in the entire completion
            boxed_count = len(re.findall(r"\\boxed\{", completion))
            
            # Check if there's exactly one \boxed{} and it appears after </think>
            if boxed_count == 1 and "\\boxed{" in after_think:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        except Exception:
            rewards.append(0.0)
    
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
        # Load dataset using TRL's DatasetMixtureConfig
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
    # Instantiate GRPO trainer
    #########################

    trainer = GRPOTrainer(
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