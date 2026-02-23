#!/usr/bin/env python3
"""
Script to merge a LoRA adapter with a base reward model.

This script loads a base reward model and a LoRA adapter, merges them together,
and saves the full merged model.
"""

import argparse
import logging
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig, PeftModel
from dotenv import load_dotenv
from huggingface_hub import HfApi

from trl.lora_utils import merge_lora_adapter
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base reward model")
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="Base model name or path (e.g., Skywork/Skywork-Reward-V2-Llama-3.1-8B)",
    )
    parser.add_argument(
        "--adapter_model_name",
        type=str,
        required=True,
        help="LoRA adapter model name or path (e.g., AmirMohseni/skywork-reward-v2-llama-3.1-8b-rank128-bard-lmarena-all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where the merged model will be saved",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "auto"],
        help="Data type for the merged model (default: bfloat16)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for loading the model (default: auto)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the merged model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face Hub model ID for pushing (if different from adapter_model_name)",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": None,
    }
    torch_dtype = dtype_map[args.dtype]

    logger.info(f"Loading base model: {args.base_model_name}")
    logger.info(f"Loading LoRA adapter: {args.adapter_model_name}")

    # Check the task type from the adapter config
    try:
        peft_config = PeftConfig.from_pretrained(args.adapter_model_name)
        logger.info(f"Detected PEFT task type: {peft_config.task_type}")
    except Exception as e:
        logger.warning(f"Could not load PEFT config: {e}. Assuming SEQ_CLS task type.")
        peft_config = None

    # Determine if this is a sequence classification (reward) model
    is_seq_cls = peft_config is None or peft_config.task_type == "SEQ_CLS"

    # Prepare model loading kwargs
    model_kwargs = {
        "device_map": args.device_map,
    }
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype

    # Load base model
    if is_seq_cls:
        logger.info("Loading as sequence classification model (reward model)")
        model_kwargs["num_labels"] = 1
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_name, **model_kwargs
        )
    else:
        logger.info("Loading as causal LM model")
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name, **model_kwargs
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_model_name)
    model.eval()

    # Merge adapter with base model using the utility function
    logger.info("Merging LoRA adapter with base model...")
    merged_model, merged_output_dir = merge_lora_adapter(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        is_main_process=True,
        torch_dtype=torch_dtype,
    )

    logger.info(f"Merged model saved to: {merged_output_dir}")

    # Optionally push to hub
    if args.push_to_hub:
        hub_model_id = args.hub_model_id or args.adapter_model_name
        logger.info(f"Pushing merged model to Hugging Face Hub: {hub_model_id}")
        
        try:
            # Push the merged model directly using push_to_hub (not from folder)
            # This will use a temporary directory automatically
            logger.info("Pushing merged model to hub...")
            merged_model.push_to_hub(hub_model_id, use_temp_dir=True)
            logger.info("Pushing tokenizer to hub...")
            tokenizer.push_to_hub(hub_model_id, use_temp_dir=True)
            logger.info(f"Successfully pushed merged model and tokenizer to {hub_model_id}")
            
            # Delete adapter files from the repository
            logger.info("Deleting adapter files from repository...")
            api = HfApi()
            
            # List all files in the repository
            try:
                repo_files = api.list_repo_files(repo_id=hub_model_id, repo_type="model")
                
                # Identify adapter files to delete
                adapter_files_to_delete = []
                adapter_file_patterns = [
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                    "adapter_weights.safetensors",
                    "adapter_weights.bin",
                ]
                
                for file in repo_files:
                    # Check if file matches any adapter pattern (handles both root and subdirectory files)
                    file_name = os.path.basename(file)
                    if any(file_name == pattern or file.endswith(f"/{pattern}") for pattern in adapter_file_patterns):
                        adapter_files_to_delete.append(file)
                
                # Delete adapter files
                if adapter_files_to_delete:
                    logger.info(f"Found {len(adapter_files_to_delete)} adapter file(s) to delete: {adapter_files_to_delete}")
                    for file_path in adapter_files_to_delete:
                        try:
                            api.delete_file(
                                path_in_repo=file_path,
                                repo_id=hub_model_id,
                                repo_type="model",
                                commit_message="Remove adapter files after merging with base model"
                            )
                            logger.info(f"Deleted adapter file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {e}")
                    logger.info("Successfully removed adapter files from repository")
                else:
                    logger.info("No adapter files found in repository to delete")
                    
            except Exception as e:
                logger.warning(f"Could not list/delete adapter files: {e}. Continuing anyway.")
                
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise


if __name__ == "__main__":
    main()

