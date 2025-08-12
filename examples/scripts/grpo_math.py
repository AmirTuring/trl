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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser, DatasetMixtureConfig, get_dataset
from math_verify import LatexExtractionConfig, parse, verify

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
class ScriptArguments:
    tokenizer_name_or_path: str = None
    dataset_seed: int = 42
    field_mapping: FieldMappingConfig = field(default_factory=FieldMappingConfig)

########################
# Helper functions
########################

def format_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on correct format: <think>...</think><answer>...</answer>
    
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
            
            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL) 
            
            # Award 1.0 if format is correct, 0.0 otherwise
            if match is not None and len(match.groups()) == 2:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    
    return rewards

def accuracy_reward(completions, target, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for completion, solution in zip(completions, target):
        try:
            # Parse the solution and completion using math_verify
            gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            answer_parsed = parse(completion, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig, dataset_args: DatasetMixtureConfig
):
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
        
        # Get training dataset (no validation needed for RL training)
        train_dataset = dataset.get("train")
        if train_dataset is None:
            raise ValueError("No training dataset found. Please ensure your dataset has a 'train' split.")
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
        
    # Generate R1-style prompt for mathematical reasoning
    def generate_math_r1_prompt(question, answer):
        """Generate R1-style prompt with step-by-step thinking format."""
        try:
            r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."
              },
              { 
                "role": "user",
                "content": f"{question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. Think step by step inside <think> tags."
              },
              {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
              }]
            
            prompt = tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True)
            return {"prompt": prompt, "target": answer}
        except Exception as e:
            logger.error(f"Error generating R1 prompt: {e}")
            # Fallback to simple format
            return {"prompt": f"Solve: {question}\nThink step by step.\n<think>", "target": answer}
    
    # Get field mapping configuration
    question_field = script_args.field_mapping.question_field
    answer_field = script_args.field_mapping.answer_field
    
    logger.info(f"Using field mapping - Question: '{question_field}', Answer: '{answer_field}'")
    
    # Convert dataset to R1 prompt format
    def extract_and_format(row):
        """Extract question and answer using configured field mapping."""
        question = row[question_field]
        answer = row[answer_field]
        
        prompt_data = generate_math_r1_prompt(question, answer)
        
        return prompt_data  # Return only {"prompt": prompt, "target": answer}
    
    logger.info("Converting dataset to R1 prompt format...")
    train_dataset = train_dataset.map(extract_and_format, desc="Processing train dataset")
    
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
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

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
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    """
    Main entry point for GRPO mathematical reasoning training.
    
    Loads configuration from YAML file and runs GRPO training with:
    - Dataset loading and field mapping
    - R1-style prompt formatting  
    - Format and accuracy reward functions
    - Step-based training with VLLM inference
    """
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args, dataset_args)


if __name__ == "__main__":
    main()