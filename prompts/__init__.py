"""
System prompts for GRPO training scripts.

This module contains system prompts used across different GRPO training configurations.
"""

# Math system prompt - used for mathematical reasoning tasks
MATH_SYSTEM_PROMPT = "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."

# VLM system prompt - used for vision-language model tasks  
THINK_SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>
This is my reasoning.
</think>
This is my answer."""

__all__ = ["MATH_SYSTEM_PROMPT", "THINK_SYSTEM_PROMPT"]
