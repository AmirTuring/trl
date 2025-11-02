FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y tmux && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system --upgrade setuptools wheel
COPY pyproject.toml uv.lock ./
RUN uv sync --system
RUN uv pip install --system flash-attn --no-build-isolation