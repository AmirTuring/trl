"""
Reward Model Training Script with Enhanced Configuration and WandB Integration

This script provides reward model training for preference learning with:
- YAML configuration file support
- WandB integration with comprehensive logging
- Multi-turn conversation support
- Field mapping for dataset flexibility (customize field names via config)
- Validation dataset support with separate field mappings
- Enhanced checkpoint handling
- LoRA support via PEFT
- Model card creation and hub integration
- Support for both pre-trained reward models and training from scratch
- Optional filtering of sequences exceeding max_length (instead of truncation)

Dataset Format:
Your dataset should have the following fields (customizable via field_mapping):

- chosen_field (default: "chosen"): The preferred conversation (single or multi-turn)
- rejected_field (default: "rejected"): The less preferred conversation (single or multi-turn)

The chosen and rejected fields should contain conversations in one of these formats:
1. List of message dicts: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
   - Supports multi-turn conversations with multiple user/assistant exchanges
2. Pre-formatted text string: Already formatted conversation text

The script automatically applies the chat template if the data is in conversation format.

Sequence Length Handling:
- By default, sequences exceeding max_length are truncated
- Set filter_long_sequences=True to discard examples where either chosen or rejected exceed max_length
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from accelerate import logging as accelerate_logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments as TrlScriptArguments,
    RewardConfig,
    RewardTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

import wandb

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class FieldMappingConfig:
    """Configuration for mapping dataset fields to expected format."""
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"

@dataclass
class ScriptArguments(TrlScriptArguments):
    tokenizer_name_or_path: str = None
    dataset_seed: int = 42
    field_mapping: FieldMappingConfig = field(default_factory=FieldMappingConfig)
    validation_field_mapping: FieldMappingConfig = field(default_factory=lambda: None)
    validation_datasets: List[Dict[str, Any]] = field(default_factory=list)
    project_name: str = field(default="reward-model-training")
    num_labels: int = field(
        default=1,
        metadata={
            "help": "Number of labels for the reward model. Use 1 for scalar rewards (default). "
        }
    )
    filter_long_sequences: bool = field(
        default=False,
        metadata={
            "help": "If True, discard examples where chosen or rejected sequences exceed max_length. "
                    "If False (default), sequences will be truncated."
        }
    )

def get_checkpoint(training_args: RewardConfig):
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    return last_checkpoint

def reward_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: RewardConfig, dataset_args: DatasetMixtureConfig):
    is_main_process = training_args.local_rank in [-1, 0]

    if not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"

    if hasattr(training_args, "report_to") and "wandb" in training_args.report_to and is_main_process:
        wandb_config = {
            "model": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs,
            "max_length": getattr(training_args, 'max_length', 1024),
            "total_batch_size": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size,
            "world_size": training_args.world_size,
            "center_rewards_coefficient": training_args.center_rewards_coefficient,
        }

        run_name = getattr(training_args, "run_name", "reward-training")
        project_name = script_args.project_name or "reward-model-training"

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=["reward-model", "preference-learning"],
            settings=wandb.Settings(start_method="fork"),
        )
        logger.info(f"W&B initialized on main process (world_size: {training_args.world_size})")
    else:
        logger.info(f"W&B disabled on rank {training_args.local_rank}")

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        use_cache=False if training_args.gradient_checkpointing else True,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        (script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path),
        revision=model_args.model_revision,
        use_fast=True,
    )
    
    # Load model with sequence classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, 
        num_labels=script_args.num_labels,
        **model_kwargs
    )
    
    logger.info(f"Loaded model with num_labels={script_args.num_labels} (1 = scalar reward score)")
    
    # Align padding tokens between tokenizer and model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        logger.warning(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs. "
            "Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )

    def format_dataset(row, field_mapping: FieldMappingConfig):
        """
        Format dataset row into chosen/rejected format for reward modeling.
        
        Supports multi-turn conversations where chosen/rejected are lists of message dicts:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        
        Also handles pre-formatted text strings.
        """
        chosen_response = row[field_mapping.chosen_field]
        rejected_response = row[field_mapping.rejected_field]
        
        # Check if chosen/rejected are in conversation format (list of dicts with role/content)
        # This supports both single-turn and multi-turn conversations
        if isinstance(chosen_response, list) and len(chosen_response) > 0 and isinstance(chosen_response[0], dict):
            # Conversation format (single or multi-turn), apply chat template
            chosen_text = tokenizer.apply_chat_template(chosen_response, tokenize=False, add_generation_prompt=False)
        else:
            # Pre-formatted text string
            chosen_text = chosen_response
        
        if isinstance(rejected_response, list) and len(rejected_response) > 0 and isinstance(rejected_response[0], dict):
            # Conversation format (single or multi-turn), apply chat template
            rejected_text = tokenizer.apply_chat_template(rejected_response, tokenize=False, add_generation_prompt=False)
        else:
            # Pre-formatted text string
            rejected_text = rejected_response
        
        return {
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    logger.info("Loading dataset...")
    try:
        if dataset_args.datasets and script_args.dataset_name:
            logger.warning("Both `datasets` and `dataset_name` provided. Using `datasets`.")
            dataset = get_dataset(dataset_args)
        elif dataset_args.datasets and not script_args.dataset_name:
            dataset = get_dataset(dataset_args)
        elif not dataset_args.datasets and script_args.dataset_name:
            dataset = load_dataset(
                script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
            )
        else:
            raise ValueError("Either `datasets` or `dataset_name` must be provided.")

        train_dataset = dataset.get(script_args.dataset_train_split) if isinstance(dataset, dict) else dataset[script_args.dataset_train_split]
        if train_dataset is None:
            available_splits = list(dataset.keys()) if isinstance(dataset, dict) else dataset.keys()
            raise ValueError(f"No dataset found for split '{script_args.dataset_train_split}'. Available splits: {available_splits}")

        logger.info(f"Training dataset size: {len(train_dataset)}")

        eval_dataset = None
        if training_args.eval_strategy != "no":
            if script_args.validation_datasets:
                from trl.scripts.utils import DatasetConfig
                validation_dataset_configs = [DatasetConfig(**val_dataset) for val_dataset in script_args.validation_datasets]
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
                    if available_splits:
                        eval_dataset = validation_dataset_dict[available_splits[0]]
                        logger.info(f"Using split '{available_splits[0]}' for validation: {len(eval_dataset)}")
            else:
                eval_dataset = dataset.get(script_args.dataset_test_split) if isinstance(dataset, dict) else dataset.get(script_args.dataset_test_split)
                if eval_dataset is not None:
                    logger.info(f"Validation dataset size: {len(eval_dataset)}")
                else:
                    logger.info("Training will proceed without validation.")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Coerce field mappings if they were parsed from YAML as dicts
    if isinstance(script_args.field_mapping, dict):
        script_args.field_mapping = FieldMappingConfig(**script_args.field_mapping)
    if script_args.validation_field_mapping is not None and isinstance(script_args.validation_field_mapping, dict):
        script_args.validation_field_mapping = FieldMappingConfig(**script_args.validation_field_mapping)

    train_field_mapping = script_args.field_mapping
    
    logger.info(f"Using field mapping - Chosen: '{train_field_mapping.chosen_field}', "
               f"Rejected: '{train_field_mapping.rejected_field}'")

    eval_field_mapping = train_field_mapping
    if script_args.validation_field_mapping is not None:
        eval_field_mapping = script_args.validation_field_mapping
        logger.info(f"Using separate validation field mapping - Chosen: '{eval_field_mapping.chosen_field}', "
                   f"Rejected: '{eval_field_mapping.rejected_field}'")

    logger.info("Formatting dataset...")
    train_dataset = train_dataset.map(
        lambda row: format_dataset(row, train_field_mapping),
        desc="Formatting train dataset"
    )
    
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda row: format_dataset(row, eval_field_mapping),
            desc="Formatting eval dataset"
        )

    # Filter long sequences if requested
    if script_args.filter_long_sequences:
        max_length = getattr(training_args, 'max_length', 1024)
        logger.info(f"Filtering sequences longer than {max_length} tokens...")
        
        def is_within_length(example):
            """Check if both chosen and rejected fit within max_length."""
            chosen_tokens = tokenizer(example['chosen'], truncation=False, add_special_tokens=True)
            rejected_tokens = tokenizer(example['rejected'], truncation=False, add_special_tokens=True)
            return len(chosen_tokens['input_ids']) <= max_length and len(rejected_tokens['input_ids']) <= max_length
        
        train_size_before = len(train_dataset)
        train_dataset = train_dataset.filter(is_within_length, desc="Filtering long sequences from train dataset")
        train_size_after = len(train_dataset)
        logger.info(f"Training dataset: {train_size_before} -> {train_size_after} examples "
                   f"({train_size_before - train_size_after} filtered, "
                   f"{100 * train_size_after / train_size_before:.2f}% retained)")
        
        if eval_dataset is not None:
            eval_size_before = len(eval_dataset)
            eval_dataset = eval_dataset.filter(is_within_length, desc="Filtering long sequences from eval dataset")
            eval_size_after = len(eval_dataset)
            logger.info(f"Validation dataset: {eval_size_before} -> {eval_size_after} examples "
                       f"({eval_size_before - eval_size_after} filtered, "
                       f"{100 * eval_size_after / eval_size_before:.2f}% retained)")

    sample = train_dataset[0]
    logger.info(f"Sample chosen text (first 300 chars): {sample['chosen'][:300]}...")
    logger.info(f"Sample rejected text (first 300 chars): {sample['rejected'][:300]}...")

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args)
    )

    # Check for checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    train_info = f"{training_args.max_steps} steps" if training_args.max_steps > 0 else f"{training_args.num_train_epochs} epochs"
    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {train_info} ***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model and tokenizer saved to {training_args.output_dir}")

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["reward-model", "preference-learning"]})

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name, commit_message=f"Reward Model checkpoint - Step {trainer.state.global_step}")
        logger.info(f"🤗 Model pushed to Hub: https://huggingface.co/{trainer.hub_model_id}")

    if trainer.accelerator.is_main_process and wandb.run:
        wandb.finish()
        logger.info("W&B logging finished")

    logger.info("*** All tasks complete! ***")

def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, RewardConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("reward", help="Run the Reward Model training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, RewardConfig, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()
    reward_function(model_args, script_args, training_args, dataset_args)

if __name__ == "__main__":
    main()

