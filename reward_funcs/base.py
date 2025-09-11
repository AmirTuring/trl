"""
Base reward function class for GRPO Math Training.

This module defines the abstract base class that all reward functions must inherit from.
It ensures a consistent interface while allowing for flexible implementations.
"""

import os
import re
import pickle
import json
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Union
from pathlib import Path


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
    
    def save(self, path: Union[str, Path], format: str = 'pickle') -> None:
        """
        Save the reward function to disk.
        
        Args:
            path: Path to save the reward function to
            format: Serialization format ('pickle' or 'json'). Default is 'pickle'.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'json':
            state = self.get_state()
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")
    
    @classmethod
    def load(cls, path: Union[str, Path], format: str = 'pickle') -> 'BaseRewardFunction':
        """
        Load a reward function from disk.
        
        Args:
            path: Path to load the reward function from
            format: Serialization format ('pickle' or 'json'). Default is 'pickle'.
            
        Returns:
            Loaded reward function instance
        """
        path = Path(path)
        
        if format == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(path, 'r') as f:
                state = json.load(f)
            return cls.from_state(state)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the serializable state of the reward function.
        
        This method should be overridden by subclasses to include their specific state.
        
        Returns:
            Dictionary containing the state of the reward function
        """
        return {
            'class_name': self.__class__.__name__,
            'module_name': self.__class__.__module__,
        }
    
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'BaseRewardFunction':
        """
        Create a reward function instance from a serialized state.
        
        This method should be overridden by subclasses to reconstruct their specific state.
        
        Args:
            state: Dictionary containing the state of the reward function
            
        Returns:
            Reconstructed reward function instance
        """
        # Default implementation for base class
        return cls()
    
    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        return self.get_state()
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        # This will be called by pickle, but we need the class to handle reconstruction
        # The actual reconstruction should be done through from_state
        pass


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
