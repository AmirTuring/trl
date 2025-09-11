"""
Reward Functions for GRPO Math Training

This module contains reward functions used for training mathematical reasoning models.
All reward functions inherit from BaseRewardFunction and maintain compatibility with
the existing GRPO trainer interface.
"""

from .base import BaseRewardFunction
from .format_reward import FormatReward
from .accuracy_reward import AccuracyReward, LLMJudgeEvaluator
from .serialization import (
    serialize_reward_function,
    deserialize_reward_function,
    create_serializable_reward_wrapper,
    make_trl_rewards_serializable
)

# For backward compatibility with existing function-based interface
def format_reward_func(completions, target, **kwargs):
    """Backward compatibility wrapper for FormatReward class."""
    reward_fn = FormatReward()
    return reward_fn(completions=completions, target=target, **kwargs)

def accuracy_reward_func(completions, target, num_generations: int = 1, **kwargs):
    """Backward compatibility wrapper for AccuracyReward class."""
    reward_fn = AccuracyReward.with_math_verify_only()
    return reward_fn(completions=completions, target=target, num_generations=num_generations, **kwargs)

__all__ = [
    'BaseRewardFunction',
    'FormatReward', 
    'AccuracyReward',
    'LLMJudgeEvaluator',
    'format_reward_func',
    'accuracy_reward_func',
    'serialize_reward_function',
    'deserialize_reward_function',
    'create_serializable_reward_wrapper',
    'make_trl_rewards_serializable'
]
