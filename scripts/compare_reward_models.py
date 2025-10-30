"""
Compare Base Model vs Fine-tuned Model for Reward Modeling

This script evaluates and compares a base reward model against a fine-tuned version
(either LoRA or full fine-tuned) on a given dataset using the same config as training.

Usage:
    python scripts/compare_reward_models.py \
        --config examples/cli_configs/reward_lora_config.yaml \
        --finetuned_model "runs/skywork-reward-v2-llama-3.1-8b-lora" \
        --dataset_split "test" \
        --output_file "reward_comparison.csv"
        
    # Or use a model from HuggingFace Hub:
    python scripts/compare_reward_models.py \
        --config examples/cli_configs/reward_lora_config.yaml \
        --finetuned_model "username/my-finetuned-reward-model" \
        --dataset_split "test" \
        --output_file "reward_comparison.csv"
        
    # Or specify a specific model revision:
    python scripts/compare_reward_models.py \
        --config examples/cli_configs/reward_lora_config.yaml \
        --finetuned_model "AmirMohseni/skywork-reward-v2-llama-3.1-8b-lora-rank16" \
        --dataset_split "test" \
        --output_file "reward_comparison.csv" \
        --model_revision "f7f831b7afd938ee86ee472db5775e46755a2226"
        
    # Or use a different validation dataset:
    python scripts/compare_reward_models.py \
        --config examples/cli_configs/reward_lora_config.yaml \
        --finetuned_model "AmirMohseni/skywork-reward-v2-llama-3.1-8b-lora-rank16" \
        --dataset_path "AmirMohseni/arena-preference-data-filtered" \
        --dataset_split "validation" \
        --output_file "reward_comparison.csv"
        
    # Or specify custom batch size for evaluation:
    python scripts/compare_reward_models.py \
        --config examples/cli_configs/reward_lora_config.yaml \
        --finetuned_model "AmirMohseni/skywork-reward-v2-llama-3.1-8b-lora-rank16" \
        --dataset_split "test" \
        --batch_size 8 \
        --output_file "reward_comparison.csv"
"""

import argparse
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import sys
from pathlib import Path
import os
from dotenv import load_dotenv # Added for .env support

# Add parent directory to path to import trl
sys.path.insert(0, str(Path(__file__).parent.parent))

from trl import ModelConfig, RewardConfig, TrlParser, get_quantization_config, get_kbit_device_map


def is_local_path(model_path):
    """Check if the model path is a local directory."""
    return os.path.exists(model_path) and os.path.isdir(model_path)


def is_lora_model(model_path):
    """Check if model_path is a LoRA adapter."""
    is_local = is_local_path(model_path)
    if is_local:
        # Check for adapter_config.json in local directory
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        return os.path.exists(adapter_config_path)
    else:
        # For HuggingFace Hub, try to check if adapter_config.json exists
        try:
            hf_hub_download(repo_id=model_path, filename="adapter_config.json")
            return True
        except:
            return False


def load_model_and_tokenizer(model_path, model_args, attn_implementation=None, num_labels=1, is_finetuned=False, model_revision=None):
    """Load model and tokenizer using the same config as training."""
    is_local = is_local_path(model_path)
    model_type = "fine-tuned" if is_finetuned else "base"
    location_type = "local" if is_local else "HuggingFace Hub"
    
    print(f"Loading {model_type} model from {location_type}: {model_path}")
    
    # Use provided model_revision if available, otherwise fall back to model_args
    revision = model_revision if model_revision is not None else model_args.model_revision
    
    if revision and revision != 'main':
        print(f"  Using revision: {revision}")
    
    # Check if fine-tuned model is a LoRA adapter
    is_lora = is_finetuned and is_lora_model(model_path)
    
    if is_lora:
        print(f"  Detected LoRA adapter model")
        # For LoRA models, we need to load the base model first
        base_model_path = model_args.model_name_or_path
        print(f"  Loading base model: {base_model_path}")
        
        # Prepare model kwargs for base model
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        
        base_model_kwargs = {
            "revision": model_args.model_revision,
            "attn_implementation": attn_implementation,
            "dtype": dtype,
            "num_labels": num_labels,
        }
        
        # Add quantization config for base model
        quantization_config = get_quantization_config(model_args)
        if quantization_config is not None:
            base_model_kwargs["device_map"] = get_kbit_device_map()
            base_model_kwargs["quantization_config"] = quantization_config
        else:
            base_model_kwargs["device_map"] = "auto"
        
        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(base_model_path, **base_model_kwargs)
        
        # Load LoRA adapter with specified revision
        print(f"  Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path, revision=revision)
        model = model.merge_and_unload()  # Merge adapter into base model for faster inference
        model.eval()
        
        # Load tokenizer from base model (adapters typically don't modify tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, revision=model_args.model_revision)
    else:
        # Regular model loading (not LoRA)
        # Prepare model kwargs
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        
        model_kwargs = {
            "revision": revision,
            "attn_implementation": attn_implementation,
            "dtype": dtype,
            "num_labels": num_labels,
        }
        
        # Add quantization config for base model only (fine-tuned already has it baked in if saved merged)
        # Also skip quantization for local fine-tuned models as they might already be quantized
        if not is_finetuned or not is_local:
            quantization_config = get_quantization_config(model_args)
            if quantization_config is not None:
                model_kwargs["device_map"] = get_kbit_device_map()
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def get_reward_scores_batch(model, tokenizer, texts, max_length=2048):
    """Get reward scores for a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Convert to float32 to avoid bf16 issues with numpy conversion
        rewards = outputs.logits[:, 0].float().cpu().numpy()
    
    return rewards


def format_conversation(example, tokenizer, field_mapping=None):
    """Format example into chosen/rejected texts."""
    if field_mapping is None:
        field_mapping = {
            "chosen_field": "chosen",
            "rejected_field": "rejected"
        }
    
    chosen = example[field_mapping["chosen_field"]]
    rejected = example[field_mapping["rejected_field"]]
    
    # Handle conversation format
    if isinstance(chosen, list) and len(chosen) > 0 and isinstance(chosen[0], dict):
        chosen_text = tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
    else:
        chosen_text = chosen
    
    if isinstance(rejected, list) and len(rejected) > 0 and isinstance(rejected[0], dict):
        rejected_text = tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
    else:
        rejected_text = rejected
    
    return chosen_text, rejected_text


def evaluate_and_compare(base_model, finetuned_model, tokenizer, dataset, field_mapping, max_length=2048, max_samples=None, batch_size=4):
    """Evaluate both models and compare results using batched inference."""
    results = []
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Process in batches
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))
        
        chosen_texts = []
        rejected_texts = []
        original_chosen_texts = []
        original_rejected_texts = []
        batch_indices = []
        
        # Prepare batch data
        for idx, example in enumerate(batch):
            try:
                # Get original texts from dataset (before tokenizer formatting)
                chosen_field = field_mapping.get("chosen_field", "chosen")
                rejected_field = field_mapping.get("rejected_field", "rejected")
                original_chosen = example[chosen_field]
                original_rejected = example[rejected_field]
                
                # Convert to string representation if it's a list/dict (for display purposes)
                if isinstance(original_chosen, (list, dict)):
                    original_chosen_str = str(original_chosen)
                else:
                    original_chosen_str = original_chosen
                
                if isinstance(original_rejected, (list, dict)):
                    original_rejected_str = str(original_rejected)
                else:
                    original_rejected_str = original_rejected
                
                # Get formatted texts for model evaluation
                chosen_text, rejected_text = format_conversation(example, tokenizer, field_mapping)
                
                chosen_texts.append(chosen_text)
                rejected_texts.append(rejected_text)
                original_chosen_texts.append(original_chosen_str)
                original_rejected_texts.append(original_rejected_str)
                batch_indices.append(batch_start + idx)
            except Exception as e:
                print(f"Error processing example {batch_start + idx}: {e}")
                continue
        
        if not chosen_texts:
            continue
        
        try:
            # Get base model scores for batch
            base_chosen_rewards = get_reward_scores_batch(base_model, tokenizer, chosen_texts, max_length)
            base_rejected_rewards = get_reward_scores_batch(base_model, tokenizer, rejected_texts, max_length)
            
            # Get finetuned model scores for batch
            ft_chosen_rewards = get_reward_scores_batch(finetuned_model, tokenizer, chosen_texts, max_length)
            ft_rejected_rewards = get_reward_scores_batch(finetuned_model, tokenizer, rejected_texts, max_length)
            
            # Process batch results
            for i, idx in enumerate(batch_indices):
                base_chosen_reward = float(base_chosen_rewards[i])
                base_rejected_reward = float(base_rejected_rewards[i])
                base_diff = base_chosen_reward - base_rejected_reward
                base_correct = 1 if base_chosen_reward >= base_rejected_reward else 0
                
                ft_chosen_reward = float(ft_chosen_rewards[i])
                ft_rejected_reward = float(ft_rejected_rewards[i])
                ft_diff = ft_chosen_reward - ft_rejected_reward
                ft_correct = 1 if ft_chosen_reward >= ft_rejected_reward else 0
                
                # Use original dataset texts (truncated for display)
                original_chosen = original_chosen_texts[i]
                original_rejected = original_rejected_texts[i]
                
                results.append({
                    "index": idx,
                    "chosen_text": original_chosen,
                    "rejected_text": original_rejected,
                    "base_reward_chosen": round(base_chosen_reward, 4),
                    "base_reward_rejected": round(base_rejected_reward, 4),
                    "base_diff": round(base_diff, 4),
                    "base_correct": base_correct,
                    "ft_reward_chosen": round(ft_chosen_reward, 4),
                    "ft_reward_rejected": round(ft_rejected_reward, 4),
                    "ft_diff": round(ft_diff, 4),
                    "ft_correct": ft_correct,
                    "improvement": 1 if ft_correct > base_correct else (-1 if ft_correct < base_correct else 0)
                })
        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")
            continue
    
    return results


def print_summary(results):
    """Print summary statistics."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    total = len(df)
    base_acc = df["base_correct"].sum() / total * 100
    ft_acc = df["ft_correct"].sum() / total * 100
    
    print(f"\nTotal samples evaluated: {total}")
    print(f"\nBase Model Accuracy: {base_acc:.2f}%")
    print(f"Fine-tuned Model Accuracy: {ft_acc:.2f}%")
    print(f"Accuracy Improvement: {ft_acc - base_acc:+.2f}%")
    
    # Breakdown by improvement
    improved = (df["improvement"] == 1).sum()
    degraded = (df["improvement"] == -1).sum()
    unchanged = (df["improvement"] == 0).sum()
    
    print(f"\nPrediction Changes:")
    print(f"  Improved (wrong → correct): {improved} ({improved/total*100:.1f}%)")
    print(f"  Degraded (correct → wrong): {degraded} ({degraded/total*100:.1f}%)")
    print(f"  Unchanged: {unchanged} ({unchanged/total*100:.1f}%)")
    
    # Average reward differences
    print(f"\nAverage Reward Differences (chosen - rejected):")
    print(f"  Base Model: {df['base_diff'].mean():.4f} (±{df['base_diff'].std():.4f})")
    print(f"  Fine-tuned Model: {df['ft_diff'].mean():.4f} (±{df['ft_diff'].std():.4f})")
    
    print("\n" + "="*80)
    
    # Show some interesting examples
    print("\nExamples where fine-tuning IMPROVED (wrong → correct):")
    improved_df = df[df["improvement"] == 1].head(3)
    for _, row in improved_df.iterrows():
        print(f"\nExample {row['index']}:")
        print(f"  Base: chosen={row['base_reward_chosen']:.3f}, rejected={row['base_reward_rejected']:.3f} ❌")
        print(f"  Fine-tuned: chosen={row['ft_reward_chosen']:.3f}, rejected={row['ft_reward_rejected']:.3f} ✅")
    
    if degraded > 0:
        print("\nExamples where fine-tuning DEGRADED (correct → wrong):")
        degraded_df = df[df["improvement"] == -1].head(3)
        for _, row in degraded_df.iterrows():
            print(f"\nExample {row['index']}:")
            print(f"  Base: chosen={row['base_reward_chosen']:.3f}, rejected={row['base_reward_rejected']:.3f} ✅")
            print(f"  Fine-tuned: chosen={row['ft_reward_chosen']:.3f}, rejected={row['ft_reward_rejected']:.3f} ❌")
    
    return df


def main():
    # Load environment variables from .env file
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") # Retrieve HF_TOKEN from environment variables
    
    # Simple argument parser without TrlParser to avoid unnecessary arguments
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned reward models")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--finetuned_model", type=str, default=None, 
                       help="Path to fine-tuned model (local directory or HuggingFace Hub model ID)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset path (e.g., 'AmirMohseni/arena-preference-data-filtered')")
    parser.add_argument("--dataset_split", type=str, default=None, help="Override dataset split (e.g., 'test')")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default="reward_comparison.csv", help="Output CSV file")
    parser.add_argument("--num_labels", type=int, default=1, help="Number of labels (1 for scalar rewards)")
    parser.add_argument("--model_revision", type=str, default=None, help="Model revision/commit hash to use for fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation (overrides config)")
    
    args = parser.parse_args()
    
    # Load the full YAML config
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Create model args from config
    class ModelArgs:
        def __init__(self, config):
            self.model_name_or_path = config.get('model_name_or_path')
            self.model_revision = config.get('model_revision', 'main')
            self.load_in_4bit = config.get('load_in_4bit', False)
            self.load_in_8bit = config.get('load_in_8bit', False)
            self.dtype = config.get('dtype', 'auto')
    
    model_args = ModelArgs(full_config)
    
    # Determine finetuned model path
    if args.finetuned_model:
        finetuned_model_path = args.finetuned_model
    else:
        # Use output_dir from config if finetuned_model not provided
        output_dir = full_config.get('output_dir')
        if not output_dir:
            raise ValueError("Either --finetuned_model must be provided or output_dir must be set in config file")
        finetuned_model_path = output_dir
    
    # Use dataset from config - prioritize validation_datasets if available
    if 'validation_datasets' in full_config and full_config['validation_datasets']:
        dataset_config = full_config['validation_datasets'][0]
        field_mapping = full_config.get('validation_field_mapping', full_config.get('field_mapping', {}))
    else:
        # Fallback to main dataset config
        dataset_config = full_config.get('datasets', [{}])[0]
        field_mapping = full_config.get('field_mapping', {})
        if not dataset_config:
            raise ValueError("No dataset configuration found in config file")
    
    # Override dataset path if provided
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = dataset_config.get('path')
    
    # Override split if provided
    if args.dataset_split:
        dataset_split = args.dataset_split
    else:
        dataset_split = dataset_config.get('split', 'test')
    
    # Get base model path from config
    base_model_path = model_args.model_name_or_path
    
    # Get num_labels from config or use provided value
    num_labels = full_config.get('num_labels', args.num_labels)
    max_length = full_config.get('max_length', 2048)
    
    attn_implementation = full_config.get('attn_implementation', None)
    
    # Get batch size - prioritize command line arg, then config, then default
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = full_config.get('per_device_eval_batch_size', 1)
    
    print(f"\nConfiguration:")
    print(f"  Base model: {base_model_path}")
    print(f"  Fine-tuned model: {finetuned_model_path}")
    if args.model_revision:
        print(f"  Fine-tuned model revision: {args.model_revision}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Split: {dataset_split}")
    print(f"  Max length: {max_length}")
    print(f"  Attn implementation: {attn_implementation}")
    print(f"  Batch size: {batch_size}")
    print(f"  Field mapping: {field_mapping}")
    print(f"  Quantization: load_in_4bit={model_args.load_in_4bit}, load_in_8bit={model_args.load_in_8bit}")
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_path} (split: {dataset_split})")
    dataset = load_dataset(dataset_path, split=dataset_split)
    print(f"Dataset size: {len(dataset)}")
    
    # Load models using config
    base_model, tokenizer = load_model_and_tokenizer(
        base_model_path, 
        model_args, 
        attn_implementation=attn_implementation,
        num_labels=num_labels,
        is_finetuned=False
    )
    finetuned_model, _ = load_model_and_tokenizer(
        finetuned_model_path,
        model_args,
        attn_implementation=attn_implementation,
        num_labels=num_labels,
        is_finetuned=True,
        model_revision=args.model_revision
    )
    
    # Evaluate and compare
    print("\nStarting evaluation...")
    results = evaluate_and_compare(
        base_model,
        finetuned_model,
        tokenizer,
        dataset,
        field_mapping,
        max_length=max_length,
        max_samples=args.max_samples,
        batch_size=batch_size
    )
    
    # Print summary and save results
    df = print_summary(results)
    df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
