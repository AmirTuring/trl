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

# Region-adjacency tree system prompt - CurveBench format (matches CurveBench-Hard)
REGION_TREE_SYSTEM_PROMPT = """Analyze this image and extract the hierarchical tree structure representing the nested regions.

The image contains nested shapes/regions. Your task is to identify the parent-child relationships between these regions.

Return the tree structure as a list of edges, where each edge is represented as (parent, child).
- The root node is always 0
- Each region is assigned a unique node number
- Edges represent parent-child relationships (a parent region contains a child region)

Format your response inside <answer>...</answer> tags.
- The first line should be the number of nodes (excluding the root).
- Each subsequent line should be "u v" meaning an edge from v to u (v is the parent, u is the child).

Example:
<answer>
3
1 0
2 0
3 1
</answer>

Think step by step about the regions and their adjacencies before providing your answer.
Format: <think>[reasoning]</think> followed by <answer>...</answer>"""

# Shape counting system prompt - for counting shapes/curves in images
REGION_COUNT_SYSTEM_PROMPT = """You are an expert at analyzing images of non-crossing closed curves and counting shapes.

Given an image of non-crossing closed curves, count the total number of shapes (closed curves) in the image.

Provide your answer as a single number inside <answer></answer> tags.
<answer>
[number]
</answer>"""

# User prompts for region tree tasks
REGION_TREE_USER_PROMPT = "Construct the region-adjacency tree for this image of non-crossing closed curves."
REGION_COUNT_USER_PROMPT = "How many shapes are in this image?"

__all__ = [
    "MATH_SYSTEM_PROMPT",
    "THINK_SYSTEM_PROMPT",
    "LLM_JUDGE_PROMPT",
    "REGION_TREE_SYSTEM_PROMPT",
    "REGION_COUNT_SYSTEM_PROMPT",
    "REGION_TREE_USER_PROMPT",
    "REGION_COUNT_USER_PROMPT",
]
