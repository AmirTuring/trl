# TRL - Transformer Reinforcement Learning

A comprehensive library to post-train foundation models using techniques like Supervised Fine-Tuning (SFT), Reinforcement Learning (RL).

## Quick Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd trl
```

### 2. Run Setup Script

```bash
sh setup.sh
```

This will:
- Install tmux (if not already installed)
- Set up a Python virtual environment
- Install all required dependencies
- Install flash-attention if CUDA is available

### 3. Configure Accelerate

Choose the appropriate configuration based on your hardware:

```bash
# For MacBook / Local Development (CPU/MPS)
cp examples/accelerate_configs/macbook_single.yaml accelerate_config.yaml

# For single GPU training
cp examples/accelerate_configs/single_gpu.yaml accelerate_config.yaml

# For multi-GPU training (8 GPUs)
cp examples/accelerate_configs/multi_gpu.yaml accelerate_config.yaml

# For DeepSpeed ZeRO Stage 2 (recommended for larger models)
cp examples/accelerate_configs/deepspeed_zero2.yaml accelerate_config.yaml
```

### 4. Auto-Shutdown (Optional - For RunPod Pods)

To save costs on cloud platforms like RunPod, you can enable automatic shutdown after GPU inactivity:

```bash
# Start the auto-shutdown monitor in background
nohup sh auto_shutdown.sh &

# To stop the auto-shutdown monitor
pkill -f auto_shutdown.sh
```

The script monitors GPU utilization and automatically stops the pod after 10 minutes of idle time (below 5% GPU usage). You can configure these thresholds by editing `auto_shutdown.sh`.

**Requirements:** `nvidia-smi`, `runpodctl`, and `RUNPOD_POD_ID` environment variable must be set.

# Usage

### GRPO Math Training

```bash
accelerate launch --config_file accelerate_config.yaml examples/scripts/grpo_math.py --config examples/cli_configs/grpo_math_config.yaml
```

### Reward Model Training

```bash
accelerate launch --config_file accelerate_config.yaml examples/scripts/reward_trainer.py --config examples/cli_configs/reward_lora_config.yaml
```

### Router Training

```bash
accelerate launch --config_file accelerate_config.yaml examples/scripts/router_trainer.py --config examples/cli_configs/router_config.yaml
```

## Key Features

- **Multiple Training Methods**: SFT, DPO, GRPO, PPO, KTO, ORPO, and more
- **Efficient Scaling**: Multi-GPU support via Accelerate and DeepSpeed
- **PEFT Integration**: LoRA/QLoRA for efficient fine-tuning on large models
- **Flash Attention**: Automatic installation when CUDA is detected

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE).
