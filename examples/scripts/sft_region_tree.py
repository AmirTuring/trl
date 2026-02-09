"""
SFT Training for Vision-Language Models on Region-Adjacency Tree Task

This script trains vision-language models using supervised fine-tuning (SFT)
to analyze images of non-crossing closed curves and generate the region-adjacency tree.

The ground truth tree from the dataset is used as the target output.

Features:
- Vision-language model support with multimodal dataset handling
- Configurable field mapping for datasets
- Separate train/validation dataset configurations
- PEFT/LoRA support
- WandB integration
- Checkpoint resumption
- Proper conversation formatting with image inputs

Usage:
    python sft_region_tree.py --config examples/cli_configs/sft_region_tree_config.yaml
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Add TRL root to Python path to access both trl and prompts
script_dir = Path(__file__).parent
trl_root = script_dir.parent.parent  # examples/scripts -> examples -> trl
sys.path.insert(0, str(trl_root))

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import wandb
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    ScriptArguments as TrlScriptArguments,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# Import prompts
from prompts import (
    REGION_TREE_SYSTEM_PROMPT,
    REGION_TREE_USER_PROMPT,
)

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
    image_field: str = "image"
    num_nodes_field: str = "num_nodes"
    tree_field: str = "tree"

@dataclass
class ScriptArguments(TrlScriptArguments):
    """Script arguments for SFT region-adjacency tree training."""
    tokenizer_name_or_path: str = None
    dataset_seed: int = 42
    field_mapping: FieldMappingConfig = field(default_factory=FieldMappingConfig)
    validation_field_mapping: FieldMappingConfig = field(default_factory=lambda: None)
    validation_datasets: List[Dict[str, Any]] = field(default_factory=list, metadata={"help": "List of validation dataset configurations"})
    project_name: str = field(default="curvebench")

    # System and user prompt overrides (optional)
    system_prompt: str = field(default=None, metadata={"help": "Override default system prompt"})
    user_prompt: str = field(default=None, metadata={"help": "Override default user prompt"})

########################
# Utilities
########################

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    return last_checkpoint

########################
# Dataset processing functions
########################

def format_tree_as_answer(tree_str: str) -> str:
    """Format a tree string into the expected answer format with think/answer tags."""
    # The tree field in the dataset is the edge list, e.g. "0 1\n1 2\n1 3"
    # We wrap it in the <answer> tags to match the expected output format
    return f"<answer>\n{tree_str}\n</answer>"


def format_example_as_conversation(row, idx, field_mapping, system_prompt, user_prompt):
    """Format a single dataset example as a multi-turn conversation for SFT.
    
    Returns a dict with 'messages' in the standard chat format that SFTTrainer expects.
    """
    try:
        image = row[field_mapping.image_field]
        tree = row.get(field_mapping.tree_field)

        if image is None:
            raise ValueError("Image is None")

        # Handle list of images - just take the first one
        if isinstance(image, list):
            if not image:
                raise ValueError("Empty image list")
            image = image[0]

        if tree is None:
            raise ValueError("Tree is None")

        # Format the tree as the assistant answer
        assistant_response = format_tree_as_answer(str(tree))

        # Build multi-turn conversation with image
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            },
        ]

        return {
            "messages": messages,
            "images": [image],
        }

    except Exception as e:
        logger.error(f"Error processing row {idx}: {e}")
        raise Exception(f"Error processing row {idx}: {e}")


########################
# Main training function
########################

def sft_region_tree_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig, dataset_args: DatasetMixtureConfig
):
    #########################
    # Initialize WandB (main process only)
    #########################
    is_main_process = training_args.local_rank in [-1, 0]

    if not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"

    if hasattr(training_args, "report_to") and "wandb" in training_args.report_to and is_main_process:
        wandb_config = {
            "model": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs,
            "max_seq_length": getattr(training_args, 'max_seq_length', 'auto'),
            "total_batch_size": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size,
            "world_size": training_args.world_size,
            "task": "sft-region-tree",
        }

        run_name = getattr(training_args, "run_name", "sft-region-tree")
        project_name = script_args.project_name or "curvebench"

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=["sft", "vlm", "region-tree"],
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
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
            padding_side="right",  # SFT typically uses right padding
        )
        logger.info(f"Successfully loaded processor from {model_args.model_name_or_path}")
    except Exception as e:
        logger.error(f"Failed to load processor from {model_args.model_name_or_path}: {e}")
        raise

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

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
    )
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    logger.info(f"Loaded model: {type(model).__name__}")

    # Apply PEFT if configured
    peft_config = get_peft_config(model_args)
    if peft_config:
        logger.info(f"Applying PEFT config: {peft_config}")
        logger.info(f"LoRA target modules: {peft_config.target_modules}")
        logger.info(f"LoRA rank: {peft_config.r}, alpha: {peft_config.lora_alpha}")
    else:
        logger.info("Using full fine-tuning")

    ###############
    # Select prompts
    ###############
    system_prompt = script_args.system_prompt or REGION_TREE_SYSTEM_PROMPT
    user_prompt = script_args.user_prompt or REGION_TREE_USER_PROMPT
    logger.info("Task: SFT for region-adjacency tree generation")

    ###############
    # Load datasets
    ###############
    logger.info("Loading dataset using YAML configuration...")
    try:
        if dataset_args.datasets:
            dataset = get_dataset(dataset_args)
        else:
            raise ValueError("No datasets specified in configuration. Please provide a YAML config with datasets.")

        train_dataset = dataset.get(script_args.dataset_train_split)
        if train_dataset is None:
            available_splits = list(dataset.keys())
            raise ValueError(f"No dataset found for split '{script_args.dataset_train_split}'. Available splits: {available_splits}")

        logger.info(f"Training dataset size: {len(train_dataset)} (using split: '{script_args.dataset_train_split}')")

        # Get validation dataset if available and evaluation is enabled
        eval_dataset = None
        if training_args.eval_strategy != "no":
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
        field_mapping = FieldMappingConfig(**field_mapping)

    logger.info(f"Using field mapping - Image: '{field_mapping.image_field}', Tree: '{field_mapping.tree_field}'")

    # Get validation field mapping (fallback to training mapping if not specified)
    val_field_mapping = script_args.validation_field_mapping or field_mapping
    if isinstance(val_field_mapping, dict):
        val_field_mapping = FieldMappingConfig(**val_field_mapping)

    ###############
    # Image validation and filtering
    ###############
    def validate_image(example):
        try:
            image = example[field_mapping.image_field]
            if isinstance(image, list):
                image = image[0] if image else None
            return image is not None
        except Exception:
            return False

    def filter_big_images(example):
        try:
            image = example[field_mapping.image_field]
            if isinstance(image, list):
                image = image[0] if image else None
            if image is None:
                return False
            return image.size[0] < 1024 and image.size[1] < 1024
        except Exception:
            return False

    def convert_to_rgb(example):
        try:
            image = example[field_mapping.image_field]
            if isinstance(image, list):
                image = image[0] if image else None
            if image is not None and image.mode != "RGB":
                image = image.convert("RGB")
            example[field_mapping.image_field] = image
            return example
        except Exception:
            return example

    def filter_has_tree(example):
        """Filter out examples without a valid tree."""
        tree = example.get(field_mapping.tree_field)
        return tree is not None and str(tree).strip() != ""

    logger.info("Validating and filtering images...")
    train_dataset = train_dataset.filter(validate_image)
    logger.info(f"Training dataset size after image validation: {len(train_dataset)}")

    train_dataset = train_dataset.filter(filter_big_images)
    logger.info(f"Training dataset size after filtering big images: {len(train_dataset)}")

    train_dataset = train_dataset.filter(filter_has_tree)
    logger.info(f"Training dataset size after filtering missing trees: {len(train_dataset)}")

    train_dataset = train_dataset.map(convert_to_rgb)

    if eval_dataset is not None:
        logger.info("Validating and filtering images in validation dataset...")
        eval_dataset = eval_dataset.filter(validate_image)
        logger.info(f"Validation dataset size after image validation: {len(eval_dataset)}")

        eval_dataset = eval_dataset.filter(filter_big_images)
        logger.info(f"Validation dataset size after filtering big images: {len(eval_dataset)}")

        eval_dataset = eval_dataset.filter(filter_has_tree)
        logger.info(f"Validation dataset size after filtering missing trees: {len(eval_dataset)}")

        eval_dataset = eval_dataset.map(convert_to_rgb)

    ###############
    # Format datasets for SFT
    ###############
    logger.info("Formatting training dataset for SFT...")

    def process_train_data(row, idx):
        return format_example_as_conversation(row, idx, field_mapping, system_prompt, user_prompt)

    train_dataset = train_dataset.map(process_train_data, with_indices=True, desc="Formatting train dataset for SFT")

    if eval_dataset is not None:
        logger.info("Formatting validation dataset for SFT...")

        def process_eval_data(row, idx):
            return format_example_as_conversation(row, idx, val_field_mapping, system_prompt, user_prompt)

        eval_dataset = eval_dataset.map(process_eval_data, with_indices=True, desc="Formatting validation dataset for SFT")

    # Verify the dataset structure
    logger.info("Verifying dataset structure...")
    sample = train_dataset[0]
    if "messages" not in sample:
        raise ValueError(f"Dataset missing 'messages' field. Available fields: {list(sample.keys())}")

    logger.info(f"Dataset validation passed. Sample has {len(sample['messages'])} messages.")
    for i, msg in enumerate(sample['messages']):
        role = msg['role']
        content_preview = str(msg['content'])[:100]
        logger.info(f"  Message {i}: role={role}, content={content_preview}...")

    ################
    # Training
    ################
    logger.info("Initializing SFTTrainer...")

    # Ensure gradient checkpointing kwargs are set
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=processor,
        peft_config=peft_config,
    )

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    train_info = f"{training_args.max_steps} steps" if training_args.max_steps > 0 else f"{training_args.num_train_epochs} epochs"
    logger.info(f'*** Starting SFT training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {train_info} ***')

    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("This might be due to memory issues, data format problems, or model compatibility issues")
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
        trainer.create_model_card({"tags": ["sft", "vlm", "region-tree"]})

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(commit_message=f"SFT Region Tree training checkpoint - Step {trainer.state.global_step}")

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
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()
    sft_region_tree_function(model_args, script_args, training_args, dataset_args)

if __name__ == "__main__":
    main()
