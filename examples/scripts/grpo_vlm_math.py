"""
GRPO Training for Vision-Language Models with Mathematical Reasoning

This script trains vision-language models using GRPO (Group Relative Policy Optimization)
on multimodal mathematical reasoning tasks. The training teaches models to solve visual 
math problems with step-by-step reasoning using <think></think> tags.

Reward functions:
1. Format reward: Ensures proper <think></think> format
2. Accuracy reward: Uses math_verify for mathematical correctness

Features:
- Advanced pass@k metrics logging (from grpo_math.py)
- Vision-language model support (from grpo_vlm.py)
- Multimodal dataset handling with images
- Configurable field mapping
- WandB integration

Usage:
    python grpo_vlm_math.py --config examples/cli_configs/grpo_vlm_math_config.yaml
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any
import functools

# Add TRL root to Python path to access both trl and reward_funcs
script_dir = Path(__file__).parent
trl_root = script_dir.parent.parent  # examples/scripts -> examples -> trl
sys.path.insert(0, str(trl_root))

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import wandb
from transformers import AutoTokenizer, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    GRPOConfig,
    ModelConfig,
    TrlParser,
    DatasetMixtureConfig,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    ScriptArguments as TrlScriptArguments,
)
from trl.trainer import IndexedGRPOTrainer
# Import reward functions from the reward_funcs module
from reward_funcs import format_reward_func, AccuracyReward, LLMJudgeConfig, FormatReward

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
    image_field: str = "image"

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
# Utilities
########################

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    return last_checkpoint

########################
# Dataset processing functions
########################

def generate_vlm_prompt(question, answer, system_prompt):
    """Generate prompt with step-by-step thinking format for VLM."""
    try:
        conversation = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}\n\nThink step-by-step inside <think>...</think> tags, then give your final answer in a boxed format (VERY IMPORTANT) \\boxed{{...}}."},
                ],
            },
        ]
        
        return {"prompt": conversation, "target": answer}
    except Exception as e:
        logger.error(f"Error generating VLM prompt: {e}")
        raise Exception(f"Error generating VLM prompt: {e}")

def extract_and_format_data(row, idx, question_field, answer_field, image_field, system_prompt):
    """Extract and format data for training/evaluation - no heavy image processing."""
    try:
        question = row[question_field]
        answer = row[answer_field]
        image = row[image_field]
        
        cleaned_question = question.replace("<image>", "").strip()
        
        # Basic image validation without processing
        if image is None:
            raise ValueError("Image is None")
        
        # Handle list of images - just take the first one
        if isinstance(image, list):
            if not image:
                raise ValueError("Empty image list")
            image = image[0]
        
        # Generate prompt data (no heavy image processing)
        prompt_data = generate_vlm_prompt(cleaned_question, answer, system_prompt)
        prompt_data["problem_id"] = int(row.get("id", idx))
        prompt_data["raw_question"] = cleaned_question
        prompt_data["raw_answer"] = answer
        prompt_data["image"] = image
        
        return prompt_data
    except Exception as e:
        logger.error(f"Error processing row {idx}: {e}")
        raise Exception(f"Error processing row {idx}: {e}")

########################
# Main training function
########################

def grpo_vlm_function(
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

        run_name = getattr(training_args, "run_name", "grpo-vlm-training")
        project_name = getattr(training_args, "project_name", "grpo-vlm-training")

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=["grpo", "vlm", "reasoning"],
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
    # Load processor for VLM
    ################
    processor = AutoProcessor.from_pretrained(
        (script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
        padding_side="left"
    )
    
    # Also get tokenizer for compatibility
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loaded processor: {type(processor).__name__}")
    logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")

    ################
    # Load and configure model
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] 
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    
    # Apply PEFT if configured
    peft_config = get_peft_config(model_args)
    if peft_config:
        logger.info(f"Applying PEFT config: {peft_config}")
        logger.info(f"LoRA target modules: {peft_config.target_modules}")
        logger.info(f"LoRA rank: {peft_config.r}, alpha: {peft_config.lora_alpha}")
    else:
        logger.info("Using full fine-tuning")
    
    # Set model init kwargs for trainer compatibility (though we pass the model directly)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

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

    # Get field mapping configuration
    field_mapping = script_args.field_mapping
    if isinstance(field_mapping, dict):
        question_field = field_mapping.get("question_field", "question")
        answer_field = field_mapping.get("answer_field", "answer")
        image_field = field_mapping.get("image_field", "image")
    else:
        question_field = field_mapping.question_field
        answer_field = field_mapping.answer_field  
        image_field = field_mapping.image_field

    logger.info(f"Using field mapping - Question: '{question_field}', Answer: '{answer_field}', Image: '{image_field}'")

    # Get validation field mapping (fallback to training mapping if not specified)
    val_field_mapping = script_args.validation_field_mapping or field_mapping
    if isinstance(val_field_mapping, dict):
        val_question_field = val_field_mapping.get("question_field", question_field)
        val_answer_field = val_field_mapping.get("answer_field", answer_field)
        val_image_field = val_field_mapping.get("image_field", image_field)
    else:
        val_question_field = val_field_mapping.question_field
        val_answer_field = val_field_mapping.answer_field
        val_image_field = val_field_mapping.image_field

    # Basic image validation without heavy processing
    def validate_image(example):
        try:
            image = example[image_field]
            if isinstance(image, list):
                image = image[0] if image else None
            return image is not None
        except Exception:
            return False

    logger.info("Validating images...")
    train_dataset = train_dataset.filter(validate_image)
    logger.info(f"Training dataset size after validation: {len(train_dataset)}")

    if eval_dataset is not None:
        logger.info("Validating images in validation dataset...")
        eval_dataset = eval_dataset.filter(lambda x: validate_image(x) )
        logger.info(f"Validation dataset size after validation: {len(eval_dataset)}")

    # Process datasets with optimal multiprocessing
    logger.info("Processing training dataset...")
    
    def process_train_data(row, idx):
        return extract_and_format_data(row, idx, question_field, answer_field, image_field, THINK_SYSTEM_PROMPT)
    
    # Disable multiprocessing to avoid PyArrow serialization issues with mixed data types
    # and prevent semaphore leaks when handling images
    train_dataset = train_dataset.map(process_train_data, with_indices=True, desc="Processing train dataset")

    # Process validation dataset if available
    if eval_dataset is not None:
        logger.info("Processing validation dataset...")
        
        def process_eval_data(row, idx):
            return extract_and_format_data(row, idx, val_question_field, val_answer_field, val_image_field, THINK_SYSTEM_PROMPT)
        
        eval_dataset = eval_dataset.map(process_eval_data, with_indices=True, desc="Processing validation dataset")

    # Verify the dataset structure
    logger.info("Verifying dataset structure...")
    sample = train_dataset[0]
    required_fields = ["prompt", "target", "problem_id", "image"]
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        logger.error(f"Missing required fields in dataset: {missing_fields}")
        logger.error(f"Available fields: {list(sample.keys())}")
        raise ValueError(f"Dataset missing required fields: {missing_fields}")

    logger.info(f"✓ Dataset validation passed. Sample prompt type: {type(sample['prompt'])}")
    logger.info(f"Sample prompt (conversation with {len(sample['prompt'])} messages): {sample['prompt']}")
    logger.info(f"Sample target: {sample['target']}")
    logger.info(f"Sample problem_id: {sample['problem_id']}")
    if sample['image'] is not None:
        logger.info(f"Sample image size: {sample['image'].size}")
    else:
        logger.info("Sample image: None")
    
    logger.info("✓ Using conversation list approach - processor will handle multimodal tokenization")

    ################
    # Training
    ################
    # Initialize reward functions with configuration
    format_reward = FormatReward()
    
    if script_args.use_llm_judge:
        # Create LLM judge configuration from script arguments
        llm_judge_config = LLMJudgeConfig(
            model_name=script_args.llm_judge_model_name,
            api_key_name=script_args.llm_judge_api_key_name,
            base_url=script_args.llm_judge_base_url,
            temperature=script_args.llm_judge_temperature
        )
        accuracy_reward = AccuracyReward.with_llm_fallback(llm_judge_config)
        logger.info("Using AccuracyReward with LLM judge fallback")
    else:
        accuracy_reward = AccuracyReward.with_math_verify_only()
        logger.info("Using AccuracyReward with math_verify only (no LLM judge)")
    
    reward_functions = [format_reward, accuracy_reward]
    
    logger.info("Initializing trainer with pre-configured model...")
    
    trainer = IndexedGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    train_info = f"{training_args.max_steps} steps" if training_args.max_steps > 0 else f"{training_args.num_train_epochs} epochs"
    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {train_info} ***')
    
    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("This might be due to memory issues, data format problems, or model compatibility issues")
        
        # Debug: Check a single batch
        logger.info("Debugging with a single training sample...")
        try:
            sample_batch = next(iter(trainer.get_train_dataloader()))
            logger.info(f"Sample batch type: {type(sample_batch)}")
            logger.info(f"Sample batch length: {len(sample_batch)}")
            
            if len(sample_batch) > 0:
                first_element = sample_batch[0]
                logger.info(f"First element type: {type(first_element)}")
                logger.info(f"First element keys: {list(first_element.keys())}")
                
                # Log specific field types
                if 'prompt' in first_element:
                    logger.info(f"Prompt type: {type(first_element['prompt'])}")
                if 'image' in first_element:
                    logger.info(f"Image type: {type(first_element['image'])}")
                if 'target' in first_element:
                    logger.info(f"Target type: {type(first_element['target'])}")
        except Exception as debug_e:
            logger.error(f"Could not debug batch: {debug_e}")
        
        raise

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
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model, tokenizer, and processor saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "vlm"]})

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")

        # Ensure model is in the same dtype as training before uploading
        if hasattr(training_args, 'torch_dtype') and training_args.torch_dtype:
            target_dtype = training_args.torch_dtype
            if hasattr(trainer.model, 'dtype') and trainer.model.dtype != target_dtype:
                logger.info(f"Converting model from {trainer.model.dtype} to {target_dtype} for hub upload")
                trainer.model.to(target_dtype)

        trainer.push_to_hub(commit_message=f"GRPO VLM training checkpoint - Step {trainer.state.global_step}")

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
    grpo_vlm_function(model_args, script_args, training_args, dataset_args)

if __name__ == "__main__":
    main()