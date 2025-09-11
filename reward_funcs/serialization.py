"""
Serialization utilities for reward functions.

This module provides utilities for serializing and deserializing reward functions,
including support for function-based rewards and dynamic class loading.
"""

import pickle
import json
import importlib
from typing import Any, Dict, Union, Callable, Optional
from pathlib import Path


def serialize_reward_function(reward_func: Union[Callable, Any], path: Union[str, Path], format: str = 'pickle') -> None:
    """
    Serialize a reward function to disk.
    
    This function can handle both class-based reward functions (with get_state method)
    and simple function-based reward functions.
    
    Args:
        reward_func: The reward function to serialize
        path: Path to save the serialized function
        format: Serialization format ('pickle' or 'json')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(reward_func, 'save'):
        # Class-based reward function with save method
        reward_func.save(path, format=format)
    elif hasattr(reward_func, 'get_state'):
        # Class-based reward function with get_state method
        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(reward_func, f)
        elif format == 'json':
            state = reward_func.get_state()
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    else:
        # Function-based reward function
        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(reward_func, f)
        elif format == 'json':
            # For function-based rewards, we store metadata about the function
            state = {
                'type': 'function',
                'module_name': getattr(reward_func, '__module__', None),
                'function_name': getattr(reward_func, '__name__', None),
                'is_closure': hasattr(reward_func, '__closure__') and reward_func.__closure__ is not None,
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


def deserialize_reward_function(path: Union[str, Path], format: str = 'pickle') -> Any:
    """
    Deserialize a reward function from disk.
    
    Args:
        path: Path to load the serialized function from
        format: Serialization format ('pickle' or 'json')
        
    Returns:
        The deserialized reward function
    """
    path = Path(path)
    
    if format == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        with open(path, 'r') as f:
            state = json.load(f)
        
        if state.get('type') == 'function':
            # Try to import the function
            module_name = state.get('module_name')
            function_name = state.get('function_name')
            
            if module_name and function_name:
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, function_name)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not import function {function_name} from {module_name}: {e}")
            else:
                raise ValueError("Function-based reward serialized to JSON cannot be fully restored without module and function name")
        else:
            # Class-based reward function
            class_name = state.get('class_name')
            module_name = state.get('module_name')
            
            if not class_name or not module_name:
                raise ValueError("Serialized state missing class_name or module_name")
            
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                return cls.from_state(state)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not import class {class_name} from {module_name}: {e}")
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_serializable_reward_wrapper(reward_func: Callable, 
                                      func_name: Optional[str] = None,
                                      module_name: Optional[str] = None) -> Any:
    """
    Create a serializable wrapper for a reward function.
    
    This is useful for making function-based rewards more easily serializable
    by providing explicit metadata.
    
    Args:
        reward_func: The reward function to wrap
        func_name: Name of the function (for JSON serialization)
        module_name: Module name of the function (for JSON serialization)
        
    Returns:
        A wrapped reward function with serialization metadata
    """
    class SerializableRewardWrapper:
        def __init__(self, func, name=None, module=None):
            self.func = func
            self.func_name = name or getattr(func, '__name__', 'unknown')
            self.module_name = module or getattr(func, '__module__', 'unknown')
        
        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
        
        def get_state(self) -> Dict[str, Any]:
            return {
                'type': 'wrapper',
                'func_name': self.func_name,
                'module_name': self.module_name,
            }
        
        @classmethod
        def from_state(cls, state: Dict[str, Any]) -> 'SerializableRewardWrapper':
            func_name = state['func_name']
            module_name = state['module_name']
            
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            
            return cls(func, func_name, module_name)
    
    return SerializableRewardWrapper(reward_func, func_name, module_name)


def make_trl_rewards_serializable():
    """
    Apply serialization wrappers to TRL reward functions.
    
    This function patches the TRL reward functions to make them serializable.
    """
    try:
        from trl.rewards import think_format_reward, get_soft_overlong_punishment
        
        # For think_format_reward, create a wrapper
        def serializable_think_format_reward(*args, **kwargs):
            return think_format_reward(*args, **kwargs)
        
        serializable_think_format_reward.__name__ = 'think_format_reward'
        serializable_think_format_reward.__module__ = 'trl.rewards.format_rewards'
        
        # For get_soft_overlong_punishment, we need to handle the closure
        def serializable_get_soft_overlong_punishment(max_completion_len: int, soft_punish_cache: int):
            """Serializable version of get_soft_overlong_punishment."""
            func = get_soft_overlong_punishment(max_completion_len, soft_punish_cache)
            
            # Create a serializable wrapper
            class SerializableSoftOverlongPunishment:
                def __init__(self, max_len, cache):
                    self.max_completion_len = max_len
                    self.soft_punish_cache = cache
                    self.func = get_soft_overlong_punishment(max_len, cache)
                
                def __call__(self, *args, **kwargs):
                    return self.func(*args, **kwargs)
                
                def get_state(self):
                    return {
                        'type': 'soft_overlong_punishment',
                        'max_completion_len': self.max_completion_len,
                        'soft_punish_cache': self.soft_punish_cache,
                    }
                
                @classmethod
                def from_state(cls, state):
                    return cls(state['max_completion_len'], state['soft_punish_cache'])
            
            return SerializableSoftOverlongPunishment(max_completion_len, soft_punish_cache)
        
        return {
            'think_format_reward': serializable_think_format_reward,
            'get_soft_overlong_punishment': serializable_get_soft_overlong_punishment,
        }
        
    except ImportError:
        # TRL rewards not available
        return {}
