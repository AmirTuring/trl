"""
SFT Training Script with Enhanced Configuration and WandB Integration

This script provides supervised fine-tuning (SFT) for language models with:
- YAML configuration file support
- WandB integration with comprehensive logging  
- Question/Completion format: Datasets should have a question field and a completion/response field
- Field mapping for dataset flexibility (customize field names via config)
- Validation dataset support with separate field mappings
- Enhanced checkpoint handling
- LoRA support via PEFT
- Model card creation and hub integration
- Proper loss masking for instruction tuning (only compute loss on assistant responses)

Dataset Format:
Your dataset should have two fields (customizable via field_mapping):
- question_field (default: "question"): The input/prompt
- completion_field (default: "response"): The expected model output/response
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from accelerate import logging as accelerate_logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments as TrlScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
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
    question_field: str = "question"
    completion_field: str = "response"

@dataclass
class ScriptArguments(TrlScriptArguments):
    tokenizer_name_or_path: str = None
    dataset_seed: int = 42
    field_mapping: FieldMappingConfig = field(default_factory=FieldMappingConfig)
    validation_field_mapping: FieldMappingConfig = field(default_factory=lambda: None)
    validation_datasets: List[Dict[str, Any]] = field(default_factory=list)
    project_name: str = field(default="sft-training")

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    return last_checkpoint

def sft_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig, dataset_args: DatasetMixtureConfig):
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
            "packing": getattr(training_args, 'packing', False),
            "total_batch_size": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size,
            "world_size": training_args.world_size,
        }

        run_name = getattr(training_args, "run_name", "sft-training")
        project_name = script_args.project_name or "sft-training"

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=["sft", "supervised-fine-tuning"],
            settings=wandb.Settings(start_method="fork"),
        )
        logger.info(f"W&B initialized on main process (world_size: {training_args.world_size})")
    else:
        logger.info(f"W&B disabled on rank {training_args.local_rank}")

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        (script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    def format_dataset(row, question_field, completion_field):
        """Format dataset row into conversational format with proper masking."""
        question = row[question_field]
        completion = row[completion_field]
        
        # Create conversation format
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": completion}
        ]
        
        # Apply chat template to get the full text
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        # For proper loss masking, we need to identify where the assistant response starts
        # First, get the prompt part (user message + assistant prefix)
        prompt_conversation = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(prompt_conversation, tokenize=False, add_generation_prompt=True)
        
        return {
            "text": text,
            "prompt": prompt_text,
            "completion": completion
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

    train_question_field = script_args.field_mapping.question_field
    train_completion_field = script_args.field_mapping.completion_field
    logger.info(f"Using field mapping - Question: '{train_question_field}', Completion: '{train_completion_field}'")

    eval_question_field = train_question_field
    eval_completion_field = train_completion_field
    if script_args.validation_field_mapping is not None:
        eval_question_field = script_args.validation_field_mapping.question_field
        eval_completion_field = script_args.validation_field_mapping.completion_field
        logger.info(f"Using validation field mapping - Question: '{eval_question_field}', Completion: '{eval_completion_field}'")

    logger.info("Formatting dataset...")
    train_dataset = train_dataset.map(
        lambda row: format_dataset(row, train_question_field, train_completion_field),
        desc="Formatting train dataset"
    )
    
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda row: format_dataset(row, eval_question_field, eval_completion_field),
            desc="Formatting eval dataset"
        )

    sample = train_dataset[0]
    logger.info(f"Sample formatted text (first 500 chars): {sample['text'][:500]}...")
    logger.info(f"Sample prompt (first 300 chars): {sample['prompt'][:300]}...")

    trainer = SFTTrainer(
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

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model and tokenizer saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["sft", "supervised-fine-tuning"]})

    if training_args.push_to_hub:
        if hasattr(training_args, 'torch_dtype') and training_args.torch_dtype:
            target_dtype = training_args.torch_dtype
            if hasattr(trainer.model, 'dtype') and trainer.model.dtype != target_dtype:
                logger.info(f"Converting model to {target_dtype} for hub upload")
                trainer.model.to(target_dtype)

        trainer.push_to_hub(dataset_name=script_args.dataset_name, commit_message=f"SFT checkpoint - Step {trainer.state.global_step}")
        logger.info(f"🤗 Model pushed to Hub: https://huggingface.co/{trainer.hub_model_id}")

    if trainer.accelerator.is_main_process and wandb.run:
        wandb.finish()
        logger.info("W&B logging finished")

    logger.info("*** All tasks complete! ***")

def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()
    sft_function(model_args, script_args, training_args, dataset_args)

if __name__ == "__main__":
    main()
