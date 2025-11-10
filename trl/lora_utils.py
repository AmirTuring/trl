"""
Utility for merging LoRA adapters with base models.
"""

import logging
import os
from typing import Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel


logger = logging.getLogger(__name__)


def merge_lora_adapter(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    is_main_process: bool = True,
    torch_dtype=None,
) -> Tuple[PreTrainedModel, str]:
    """
    Merge LoRA adapter weights with the base model and save it.
    
    This function takes a PEFT model with LoRA adapters and merges the adapter weights
    into the base model, creating a standalone model that doesn't require the adapter.
    
    Args:
        model (`PreTrainedModel`):
            The model with LoRA adapters to merge. Must be a PeftModel instance.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer associated with the model.
        output_dir (`str`):
            The directory where the merged model will be saved.
        is_main_process (`bool`, *optional*, defaults to `True`):
            Whether this is the main process (for distributed training). Only the main
            process should save the model.
        torch_dtype (`torch.dtype`, *optional*):
            The dtype to use when saving the merged model. If not provided, uses the
            model's current dtype. This prevents unwanted conversion to fp32.
    
    Returns:
        merged_model (`PreTrainedModel`):
            The merged model with adapter weights integrated into the base model.
        merged_output_dir (`str`):
            The path where the merged model was saved.
    
    Raises:
        ImportError: If PEFT is not installed.
        ValueError: If the model is not a PeftModel.
        Exception: If merging fails for any reason.
    
    Example:
    ```python
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import get_peft_model, LoraConfig
    from trl import merge_lora_adapter
    
    # After training with LoRA
    model = trainer.model  # PeftModel with LoRA
    tokenizer = trainer.tokenizer
    
    # Merge adapter with base model
    merged_model, save_path = merge_lora_adapter(
        model=model,
        tokenizer=tokenizer,
        output_dir="./runs/my-model"
    )
    
    # Now you can push merged_model to hub or use it directly
    merged_model.push_to_hub("username/my-merged-model")
    ```
    """
    if not isinstance(model, PeftModel):
        raise ValueError(
            f"Model must be a PeftModel instance to merge adapters. Got {type(model).__name__}"
        )
    
    logger.info("Starting LoRA adapter merge with base model...")
    
    # Merge the adapter weights into the base model
    try:
        merged_model = model.merge_and_unload()
        logger.info("LoRA adapter successfully merged with base model")
    except Exception as e:
        logger.error(f"Failed to merge LoRA adapter: {e}")
        raise
    
    # Create output directory for merged model
    merged_output_dir = os.path.join(output_dir, "merged_model")
    
    if is_main_process:
        os.makedirs(merged_output_dir, exist_ok=True)
        
        # Determine dtype for saving
        save_dtype = torch_dtype if torch_dtype is not None else getattr(merged_model.config, 'torch_dtype', None)
        
        # Convert model to target dtype if specified
        if save_dtype is not None:
            logger.info(f"Converting merged model to dtype: {save_dtype}")
            merged_model = merged_model.to(dtype=save_dtype)
            # Update config to reflect the dtype
            merged_model.config.torch_dtype = save_dtype
        
        # Save the merged model
        logger.info(f"Saving merged model with dtype: {merged_model.dtype if hasattr(merged_model, 'dtype') else save_dtype}")
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        
        logger.info(f"Merged model saved to {merged_output_dir}")
    
    return merged_model, merged_output_dir

