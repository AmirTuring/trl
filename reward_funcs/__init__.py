"""
Reward Functions for GRPO Math Training

This module contains reward functions used for training mathematical reasoning models.
All reward functions inherit from BaseRewardFunction and maintain compatibility with
the existing GRPO trainer interface.
"""

import os
from dotenv import load_dotenv
from .base import BaseRewardFunction
from .format_reward import FormatReward
from .accuracy_reward import AccuracyReward, LLMJudgeEvaluator, LLMJudgeConfig
from .serialization import (
    serialize_reward_function,
    deserialize_reward_function,
    create_serializable_reward_wrapper,
    make_trl_rewards_serializable
)

def get_reward_func(reward_type, config=None):
    """Get a properly configured reward function."""
    if reward_type == "format":
        def format_reward_func(completions, target, **kwargs):
            reward_fn = FormatReward()
            return reward_fn(completions=completions, target=target, **kwargs)
        return format_reward_func
    
    elif reward_type == "accuracy":
        def accuracy_reward_func(completions, target, num_generations: int = 1, **kwargs):
            if config and config.get('use_llm_judge', False):
                llm_judge_config = LLMJudgeConfig(
                    model_name=config.get('llm_judge_model_name', 'gpt-5-mini'),
                    api_key_name=config.get('llm_judge_api_key_name', 'OPENAI_API_KEY'),
                    base_url=config.get('llm_judge_base_url', None),
                    temperature=config.get('llm_judge_temperature', 0.0)
                )
                reward_fn = AccuracyReward.with_llm_fallback(llm_judge_config)
            else:
                reward_fn = AccuracyReward.with_math_verify_only()
            
            return reward_fn(completions=completions, target=target, num_generations=num_generations, **kwargs)
        return accuracy_reward_func
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

# For backward compatibility with existing function-based interface
def format_reward_func(completions, target, **kwargs):
    """Backward compatibility wrapper for FormatReward class."""
    reward_fn = FormatReward()
    return reward_fn(completions=completions, target=target, **kwargs)

def accuracy_reward_func(completions, target, num_generations: int = 1, **kwargs):
    """Backward compatibility wrapper for AccuracyReward class."""
    default_config = LLMJudgeConfig(model_name="gpt-5-mini", api_key_name="OPENAI_API_KEY")
    reward_fn = AccuracyReward.with_llm_fallback(config=default_config)
    return reward_fn(completions=completions, target=target, num_generations=num_generations, **kwargs)

__all__ = [
    'BaseRewardFunction',
    'FormatReward', 
    'AccuracyReward',
    'LLMJudgeEvaluator',
    'LLMJudgeConfig',
    'get_reward_func',
    'format_reward_func',
    'accuracy_reward_func',
    'serialize_reward_function',
    'deserialize_reward_function',
    'create_serializable_reward_wrapper',
    'make_trl_rewards_serializable'
]