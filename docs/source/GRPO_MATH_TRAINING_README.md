# GRPO Math Training Guide

This guide provides step-by-step instructions for training language models using GRPO (Group Relative Policy Optimization) on mathematical reasoning tasks.

## Overview

GRPO training teaches language models to solve math problems with step-by-step reasoning in a structured format:
- `<think>step-by-step reasoning</think>`
- `<answer>final answer</answer>`

The training uses two reward functions:
1. **Format reward**: Ensures proper `<think></think><answer></answer>` structure
2. **Accuracy reward**: Uses math_verify to evaluate mathematical correctness

### Installation

1. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables (optional but recommended):**
```bash
export WANDB_API_KEY="your_wandb_api_key"  # For experiment tracking
export HF_TOKEN="your_huggingface_token"   # For model uploads
```

## Configuration Files

### 1. Accelerate Configuration

Choose the appropriate accelerate configuration based on your hardware:

- **Single GPU**: `examples/accelerate_configs/single_gpu.yaml`
- **Multi-GPU**: `examples/accelerate_configs/multi_gpu.yaml`
- **DeepSpeed ZeRO Stage 2**: `examples/accelerate_configs/deepspeed_zero2.yaml`
- **DeepSpeed ZeRO Stage 3**: `examples/accelerate_configs/deepspeed_zero3.yaml`
- **FSDP**: `examples/accelerate_configs/fsdp1.yaml` or `fsdp2.yaml`

### 2. Training Configuration

The training configuration is defined in `examples/cli_configs/grpo_math_config.yaml`:

Key parameters:
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct` (can be changed to other models)
- **Dataset**: `AmirMohseni/Big-Math-RL-filtered`
- **Batch size**: 1 per device with 8 gradient accumulation steps
- **Learning rate**: 5.0e-7 with cosine scheduler
- **GRPO beta**: 0.001 (controls exploration vs exploitation)
- **Generation settings**: 16 generations per prompt

## Step-by-Step Training Instructions

### Step 1: Initialize Accelerate Configuration

First, set up accelerate for your hardware configuration:

```bash
accelerate config
```

Alternatively, use one of the pre-configured files:

```bash
# For single GPU training
cp examples/accelerate_configs/single_gpu.yaml accelerate_config.yaml

# For multi-GPU training (8 GPUs)
cp examples/accelerate_configs/multi_gpu.yaml accelerate_config.yaml

# For DeepSpeed ZeRO Stage 2 (recommended for larger models)
cp examples/accelerate_configs/deepspeed_zero2.yaml accelerate_config.yaml
```


### Step 2: Run Training

Execute the training command:

#### For text models
```bash
accelerate launch --config_file accelerate_config.yaml examples/scripts/grpo_math.py --config examples/cli_configs/grpo_math_config.yaml
```

#### For VLM models
```bash
accelerate launch --config_file accelerate_config.yaml examples/scripts/grpo_vlm_math.py --config examples/cli_configs/grpo_vlm_config.yaml
```

### Step 4: Monitor Training

The training will log progress to:
- **Console**: Real-time training metrics
- **WandB**: If configured, detailed experiment tracking
- **Local files**: Saved in the selected `output_dir`

Key metrics to monitor:
- `format_reward`: Should approach 1.0 as model learns proper formatting
- `accuracy_reward`: Mathematical correctness score
- `reward`: Total reward
- `completion_length`: Length of the generated completion

## Training Configuration Options

### Model Configuration
```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct  # Base model
torch_dtype: bfloat16                             # Precision
bf16: true                                        # Mixed precision training
```

### GRPO Parameters
```yaml
beta: 0.001                    # KL penalty coefficient
max_prompt_length: 512         # Maximum input length
max_completion_length: 2048    # Maximum generation length
num_generations: 16            # Generations per prompt for training
```

### VLLM Configuration (for fast inference)
```yaml
use_vllm: true                      # Enable VLLM
vllm_mode: colocate                 # Run VLLM on same GPUs
vllm_gpu_memory_utilization: 0.5    # GPU memory allocation
```

### Dataset Configuration
```yaml
datasets:
  - path: AmirMohseni/Big-Math-RL-filtered
    split: train

field_mapping:
  question_field: question      # Field containing math problems
  answer_field: answer_correct  # Field containing correct answers
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16
   
   # Enable gradient checkpointing
   gradient_checkpointing: true
   ```

2. **Dataset Loading Errors**
   ```bash
   # Check field mapping in config
   field_mapping:
     question_field: your_question_field
     answer_field: your_answer_field
   ```