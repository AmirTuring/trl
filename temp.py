from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

from dotenv import load_dotenv
load_dotenv()

# ---- 1. Base & LoRA model identifiers ----
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
lora_model_id = "AmirMohseni/qwen2.5-3b-gsm8k-cipher-lora-sft"
merged_model_id = "AmirMohseni/Qwen2.5-3b-gsm8k-cipher-merged"  # your new model name on Hub

# ---- 2. Load base + LoRA ----
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_model_id)

# ---- 3. Merge LoRA weights into base ----
print("Merging LoRA weights...")
model = model.merge_and_unload()

# ---- 4. Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(lora_model_id)

# ---- 5. Push to Hugging Face Hub ----
print("Pushing to Hugging Face Hub...")
model.push_to_hub(merged_model_id)
tokenizer.push_to_hub(merged_model_id)

print(f"✅ Successfully pushed merged model to: https://huggingface.co/{merged_model_id}")