"""
Format Reward Function for GRPO Math Training.

This module contains the FormatReward class that evaluates completions based on 
strict format requirements: <think>...</think> followed by exactly one \\boxed{...}
"""

import os
import re
import random
from typing import List

from .base import BaseRewardFunction


class FormatReward(BaseRewardFunction):
    """
    Evaluates completions based on strict format requirements.
    
    Expected format:
        <think>...nonempty reasoning...</think> followed by exactly one \\boxed{...}
    
    Rewarding scheme:
        -1.0: invalid / reward hacking attempt
         0.0: partially correct but missing constraints
         1.0: valid format
    """
    
    def __init__(self, sampling_rate: float = 0.01, min_think_length: int = 10, max_ratio: float = 4.0):
        """
        Initialize the FormatReward function.
        
        Args:
            sampling_rate: Rate at which to sample completions for debugging (default: 0.01)
            min_think_length: Minimum required length for think content (default: 10)
            max_ratio: Maximum ratio of after_think to think_content length (default: 4.0)
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.min_think_length = min_think_length
        self.max_ratio = max_ratio
    
    def calculate_rewards(self, completions: List[str], **kwargs) -> List[float]:
        """
        Calculate format rewards for a list of completions.
        
        Args:
            completions: List of text completions to evaluate
            **kwargs: Additional arguments (unused but maintained for compatibility)
            
        Returns:
            List of float reward values (-1.0, 0.0, or 1.0) for each completion
        """
        rewards = []
        
        for completion in completions:
            try:
                # Ensure consistency with prompt (prepend <think>)
                completion = "<think>" + completion
                
                # Sample completions for debugging
                if random.random() < self.sampling_rate:
                    os.makedirs("completion_samples", exist_ok=True)
                    with open("completion_samples/format_completion_samples.txt", "a") as f:
                        f.write("\n\n==============\n")
                        f.write(completion)
                
                # Check for think pattern
                think_pattern = r"^<think>(.*?)</think>"
                think_match = re.search(think_pattern, completion, re.DOTALL)
                
                if not think_match:
                    rewards.append(-1.0)
                    continue
                
                think_content = think_match.group(1).strip()
                after_think = completion[think_match.end():]
                
                # Validate think content length and no nested think tags
                if len(think_content) < self.min_think_length or "<think>" in after_think:
                    rewards.append(-1.0)
                    continue
                
                # Check for exactly one boxed answer in the right place
                boxed_matches = re.findall(r"\\boxed\{.*?\}", completion)
                if len(boxed_matches) != 1 or "\\boxed{" not in after_think:
                    rewards.append(-1.0)
                    continue
                
                # Check ratio constraint
                if len(after_think.strip()) > self.max_ratio * len(think_content):
                    rewards.append(0.0)
                    continue
                
                rewards.append(1.0)
                
            except Exception:
                rewards.append(0.0)
        
        return rewards
