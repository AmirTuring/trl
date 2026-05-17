"""
Legal Classification Training Script

This script trains sequence classification models for legal content analysis with:
- YAML configuration file support
- WandB integration with comprehensive logging
- Conversation-based input: reads a 'messages' column (list of {role, content} dicts)
- Input mode: 'full' (all messages) or 'user' (user messages only)
- Field mapping for dataset flexibility (customize field names via config)
- Validation dataset support with separate field mappings
- Enhanced checkpoint handling
- LoRA support via PEFT
- Model card creation and hub integration
- Configurable label mappings

Designed for datasets like AmirMohseni/WildChat-Legal-Classification-V2-Balanced
with targets: contains_legal_content, seeks_legal_guidance, primary_topic.

Usage:
    python legal_classifier.py --config examples/cli_configs/legal_contains_legal_config.yaml
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
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

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class LegalFieldMappingConfig:
    """Configuration for mapping dataset fields to expected format."""
    messages_field: str = "messages"
    label_field: str = "contains_legal_content"


@dataclass
class ScriptArguments(TrlScriptArguments):
    tokenizer_name_or_path: str = None
    dataset_seed: int = 42
    field_mapping: LegalFieldMappingConfig = field(default_factory=LegalFieldMappingConfig)
    validation_field_mapping: LegalFieldMappingConfig = field(default_factory=lambda: None)
    validation_datasets: List[Dict[str, Any]] = field(default_factory=list)
    project_name: str = field(default="legal-classification")
    label_mapping: Dict[str, int] = field(default_factory=lambda: {"False": 0, "True": 1})
    max_length: int = 512
    input_mode: str = field(
        default="full",
        metadata={"help": "How to serialize conversations: 'full' (all messages) or 'user' (user messages only)"},
    )


def get_checkpoint(training_args: TrainingArguments):
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    return last_checkpoint


def serialize_conversation(messages: List[Dict[str, str]], input_mode: str) -> str:
    """Serialize a conversation (list of {role, content} dicts) into a flat string.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        input_mode: 'full' to include all messages, 'user' to include only user messages.

    Returns:
        A single string representing the conversation.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if input_mode == "user" and role != "user":
            continue

        prefix = role.capitalize()
        parts.append(f"{prefix}: {content}")

    return "\n".join(parts)


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


def legal_classifier_function(
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: TrainingArguments,
    dataset_args: DatasetMixtureConfig,
):
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
            "input_mode": script_args.input_mode,
            "total_batch_size": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size,
            "world_size": training_args.world_size,
            "label_mapping": script_args.label_mapping,
        }

        run_name = getattr(training_args, "run_name", "legal-classifier")
        project_name = script_args.project_name or "legal-classification"

        wandb_token = os.getenv("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=["legal", "sequence-classification", "text-classification"],
            settings=wandb.Settings(start_method="fork"),
        )
        logger.info(f"W&B initialized on main process (world_size: {training_args.world_size})")
    else:
        logger.info(f"W&B disabled on rank {training_args.local_rank}")

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Input mode: {script_args.input_mode}")

    # Label mapping
    id2label = {v: k for k, v in script_args.label_mapping.items()}
    label2id = script_args.label_mapping
    num_labels = len(label2id)

    logger.info(f"Label mapping: {label2id}")
    logger.info(f"Number of labels: {num_labels}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
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

    def format_dataset(row, messages_field, label_field):
        """Format dataset row: serialize conversation and tokenize."""
        messages = row[messages_field]
        label = row[label_field]

        # Serialize conversation based on input_mode
        text = serialize_conversation(messages, script_args.input_mode)

        # Convert label to int
        if isinstance(label, bool):
            label = label2id[str(label)]
        elif isinstance(label, str):
            if label not in label2id:
                raise ValueError(f"Unknown label '{label}'. Expected one of: {list(label2id.keys())}")
            label = label2id[label]

        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=script_args.max_length,
            padding=False,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": label,
        }

    # Load datasets
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
        script_args.field_mapping = LegalFieldMappingConfig(**script_args.field_mapping)
    if script_args.validation_field_mapping is not None and isinstance(script_args.validation_field_mapping, dict):
        script_args.validation_field_mapping = LegalFieldMappingConfig(**script_args.validation_field_mapping)

    train_messages_field = script_args.field_mapping.messages_field
    train_label_field = script_args.field_mapping.label_field
    logger.info(f"Using field mapping - Messages: '{train_messages_field}', Label: '{train_label_field}'")

    eval_messages_field = train_messages_field
    eval_label_field = train_label_field
    if script_args.validation_field_mapping is not None:
        eval_messages_field = script_args.validation_field_mapping.messages_field
        eval_label_field = script_args.validation_field_mapping.label_field
        logger.info(f"Using validation field mapping - Messages: '{eval_messages_field}', Label: '{eval_label_field}'")

    # Format datasets
    logger.info("Formatting dataset...")
    train_dataset = train_dataset.map(
        lambda row: format_dataset(row, train_messages_field, train_label_field),
        desc="Formatting train dataset",
        remove_columns=train_dataset.column_names,
    )

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda row: format_dataset(row, eval_messages_field, eval_label_field),
            desc="Formatting eval dataset",
            remove_columns=eval_dataset.column_names,
        )

    # Log a sample
    sample = train_dataset[0]
    logger.info(f"Sample input_ids length: {len(sample['input_ids'])}")
    logger.info(f"Sample label: {sample['label']} ({id2label[sample['label']]})")
    logger.info(f"Sample text (decoded, first 200 chars): {tokenizer.decode(sample['input_ids'][:50])}...")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize trainer
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

    # Train
    train_info = f"{training_args.max_steps} steps" if training_args.max_steps > 0 else f"{training_args.num_train_epochs} epochs"
    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {train_info} ***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # Save
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model and tokenizer saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["legal", "sequence-classification", "text-classification"]})

    if training_args.push_to_hub:
        trainer.push_to_hub(
            dataset_name=script_args.dataset_name,
            commit_message=f"Legal classifier checkpoint - Step {trainer.state.global_step}",
        )

    if trainer.accelerator.is_main_process and wandb.run:
        wandb.finish()
        logger.info("W&B logging finished")

    logger.info("*** All tasks complete! ***")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, TrainingArguments, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("legal_classifier", help="Run the Legal Classifier training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, TrainingArguments, DatasetMixtureConfig))
    model_args, script_args, training_args, dataset_args = parser.parse_args_and_config()
    legal_classifier_function(model_args, script_args, training_args, dataset_args)


if __name__ == "__main__":
    main()
