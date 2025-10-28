"""
Router Training Script with Enhanced Configuration and WandB Integration

This script provides sequence classification training for router models with:
- YAML configuration file support
- WandB integration with comprehensive logging  
- Prompt/Label format: Datasets should have a prompt field and a label field
- Field mapping for dataset flexibility (customize field names via config)
- Validation dataset support with separate field mappings
- Enhanced checkpoint handling
- LoRA support via PEFT
- Model card creation and hub integration
- Configurable label mappings (e.g., {"no_web": 0, "web": 1})

Dataset Format:
Your dataset should have two fields (customizable via field_mapping):
- prompt_field (default: "prompt"): The input text
- label_field (default: "label"): The classification label (string or int)
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from accelerate import logging as accelerate_logging
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments as TrlScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class RouterFieldMappingConfig:
    """Configuration for mapping dataset fields to expected format."""
    prompt_field: str = "prompt"
    label_field: str = "label"

@dataclass
class ScriptArguments(TrlScriptArguments):
    tokenizer_name_or_path: str = None
    dataset_seed: int = 42
    field_mapping: RouterFieldMappingConfig = field(default_factory=RouterFieldMappingConfig)
    validation_field_mapping: RouterFieldMappingConfig = field(default_factory=lambda: None)
    validation_datasets: List[Dict[str, Any]] = field(default_factory=list)
    project_name: str = field(default="router-training")
    label_mapping: Dict[str, int] = field(default_factory=lambda: {"no_web": 0, "web": 1})
    max_length: int = 512

def get_checkpoint(training_args: TrainingArguments):
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    return last_checkpoint

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def router_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: TrainingArguments, dataset_args: DatasetMixtureConfig):
    is_main_process = training_args.local_rank in [-1, 0]

    if not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"

    if hasattr(training_args, "report_to") and "wandb" in training_args.report_to and is_main_process:
        wandb_config = {
            "model": model_args.model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs,
            "max_length": script_args.max_length,
            "total_batch_size": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size,
            "world_size": training_args.world_size,
            "label_mapping": script_args.label_mapping,
        }

        run_name = getattr(training_args, "run_name", "router-training")
        project_name = script_args.project_name or "router-training"

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=["router", "sequence-classification"],
            settings=wandb.Settings(start_method="fork"),
        )
        logger.info(f"W&B initialized on main process (world_size: {training_args.world_size})")
    else:
        logger.info(f"W&B disabled on rank {training_args.local_rank}")

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Create reverse label mapping (int -> string) for verification
    id2label = {v: k for k, v in script_args.label_mapping.items()}
    label2id = script_args.label_mapping
    num_labels = len(label2id)
    
    logger.info(f"Label mapping: {label2id}")
    logger.info(f"Number of labels: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(
        (script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config first to check if model supports use_cache
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    
    model_kwargs = dict(
        config=config,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, 
        **model_kwargs
    )

    # Apply PEFT if configured
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        from peft import get_peft_model
        logger.info(f"Applying PEFT with config: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    def format_dataset(row, prompt_field, label_field):
        """Format dataset row for sequence classification."""
        prompt = row[prompt_field]
        label = row[label_field]
        
        # Convert string label to int if necessary
        if isinstance(label, str):
            if label not in label2id:
                raise ValueError(f"Unknown label '{label}'. Expected one of: {list(label2id.keys())}")
            label = label2id[label]
        
        # Tokenize the prompt
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=script_args.max_length,
            padding=False,  # We'll pad in the collator
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": label,
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
        script_args.field_mapping = RouterFieldMappingConfig(**script_args.field_mapping)
    if script_args.validation_field_mapping is not None and isinstance(script_args.validation_field_mapping, dict):
        script_args.validation_field_mapping = RouterFieldMappingConfig(**script_args.validation_field_mapping)

    train_prompt_field = script_args.field_mapping.prompt_field
    train_label_field = script_args.field_mapping.label_field
    logger.info(f"Using field mapping - Prompt: '{train_prompt_field}', Label: '{train_label_field}'")

    eval_prompt_field = train_prompt_field
    eval_label_field = train_label_field
    if script_args.validation_field_mapping is not None:
        eval_prompt_field = script_args.validation_field_mapping.prompt_field
        eval_label_field = script_args.validation_field_mapping.label_field
        logger.info(f"Using validation field mapping - Prompt: '{eval_prompt_field}', Label: '{eval_label_field}'")

    logger.info("Formatting dataset...")
    train_dataset = train_dataset.map(
        lambda row: format_dataset(row, train_prompt_field, train_label_field),
        desc="Formatting train dataset",
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda row: format_dataset(row, eval_prompt_field, eval_label_field),
            desc="Formatting eval dataset",
            remove_columns=eval_dataset.column_names,
        )

    sample = train_dataset[0]
    logger.info(f"Sample input_ids length: {len(sample['input_ids'])}")
    logger.info(f"Sample label: {sample['label']} ({id2label[sample['label']]})")
    logger.info(f"Sample text (decoded): {tokenizer.decode(sample['input_ids'][:100])}...")

    # Create data collator for padding
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
        trainer.create_model_card({"tags": ["router", "sequence-classification", "text-classification"]})

    if training_args.push_to_hub:
        trainer.push_to_hub(commit_message=f"Router checkpoint - Step {trainer.state.global_step}")
        logger.info(f"🤗 Model pushed to Hub: https://huggingface.co/{trainer.hub_model_id}")

    if trainer.accelerator.is_main_process and wandb.run:
        wandb.finish()
        logger.info("W&B logging finished")

    logger.info("*** All tasks complete! ***")

def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, TrainingArguments, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("router", help="Run the Router training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, TrainingArguments, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()
    router_function(model_args, script_args, training_args, dataset_args)

if __name__ == "__main__":
    main()

