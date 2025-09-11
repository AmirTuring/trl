"""
Base reward function class for GRPO Math Training.

This module defines the abstract base class that all reward functions must inherit from.
It ensures a consistent interface while allowing for flexible implementations.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import List, Any, Dict


class BaseRewardFunction(ABC):
    """
    Abstract base class for all reward functions.
    
    All reward functions must implement the calculate_rewards method that takes
    completions and optional kwargs and returns a list of reward values.
    """
    
    def __init__(self):
        """Initialize the reward function."""
        pass
    
    @abstractmethod
    def calculate_rewards(self, completions: List[str], **kwargs) -> List[float]:
        """
        Calculate rewards for a list of completions.
        
        Args:
            completions: List of text completions to evaluate
            **kwargs: Additional keyword arguments that may be needed by specific reward functions
            
        Returns:
            List of float reward values, one for each completion
        """
        pass
    
    def __call__(self, completions: List[str] = None, **kwargs) -> List[float]:
        """
        Make the reward function callable with the same interface as the original functions.
        
        This maintains backward compatibility with the existing GRPO trainer interface.
        """
        if completions is None:
            completions = kwargs.get('completions', [])
        
        return self.calculate_rewards(completions, **kwargs)


def _extract_boxed_answer(completion: str) -> str:
    """Extract answer from \\boxed{} format, handling nested braces."""
    if "\\boxed{" not in completion:
        return completion
    start_idx = completion.find("\\boxed{") + 7
    brace_count = 1
    end_idx = start_idx
    while end_idx < len(completion) and brace_count > 0:
        if completion[end_idx] == "{":
            brace_count += 1
        elif completion[end_idx] == "}":
            brace_count -= 1
        end_idx += 1
    return completion[start_idx:end_idx - 1].strip()


def _select_for_index(arr, i, G, total_len):
    """Robustly select arr item for completion index i."""
    if isinstance(arr, (list, tuple)):
        if len(arr) == total_len:
            return arr[i]
        if len(arr) * G == total_len:
            return arr[i // max(1, G)]
        if len(arr) == 1:
            return arr[0]
        # fallback
        return arr[min(i, len(arr) - 1)]
    return arr
