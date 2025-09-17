"""
System prompts for GRPO training scripts.

This module contains system prompts used across different GRPO training configurations.
"""

# Math system prompt - used for mathematical reasoning tasks
MATH_SYSTEM_PROMPT = "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."

# VLM system prompt - used for vision-language model tasks  
THINK_SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think></think> tags, and the final answer is provided inside \\boxed{} format, i.e., <think>
This is my reasoning.
</think>
The answer is \\boxed{final answer}.
"""

# LLM Judge prompt - used for mathematical answer evaluation
LLM_JUDGE_PROMPT = """You are an expert evaluator. Your task is to determine if two answers are equivalent.
Ground Truth Answer: {ground_truth}
Student Answer: {completion_answer}
Compare these two answers and determine if they are equivalent. Consider:
- Numerical equivalence (e.g., 0.5 = 1/2)
- Algebraic equivalence (e.g., x^2 - 1 = (x-1)(x+1))
- Different valid forms of the same answer
- Rounding differences within very small and reasonable tolerance
Return a correctness score where 1.0 means the answers are equivalent and 0.0 means they are not equivalent. If no answer was given, return 0.0."""

__all__ = ["MATH_SYSTEM_PROMPT", "THINK_SYSTEM_PROMPT", "LLM_JUDGE_PROMPT"]
