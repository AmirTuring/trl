"""
Accuracy Reward Function for GRPO Math Training.

This module contains the AccuracyReward class that evaluates mathematical correctness
using math_verify with optional LLM-as-a-judge fallback when math_verify fails or returns 0.
"""

import logging
from typing import List, Any, Optional, Dict
from abc import abstractmethod

from .base import BaseRewardFunction, _extract_boxed_answer, _select_for_index
from math_verify import math_metric, LatexExtractionConfig, ExprExtractionConfig

logger = logging.getLogger(__name__)


class LLMJudgeEvaluator:
    """
    LLM-as-a-judge evaluator for mathematical reasoning.
    
    This class provides LLM-based evaluation as a fallback when math_verify
    fails or returns 0.
    """
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LLM judge evaluator.
        
        Args:
            model_name: Name of the LLM to use for evaluation
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.api_key = api_key
        # TODO: Initialize LLM client (OpenAI, Anthropic, etc.)
        
    def evaluate(self, completion_answer: str, ground_truth: str) -> float:
        """
        Evaluate using LLM as a judge.
        
        This should be implemented to:
        1. Format a prompt for the LLM judge asking it to compare answers
        2. Call the LLM API
        3. Parse the response to get a score
        
        Args:
            completion_answer: The extracted answer from the completion  
            ground_truth: The ground truth answer
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # TODO: Implement LLM-based evaluation
        # For now, return 0.0 as placeholder
        return 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get the serializable state of the LLM judge evaluator."""
        return {
            'model_name': self.model_name,
            'api_key': self.api_key,
        }
    
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'LLMJudgeEvaluator':
        """Create an LLMJudgeEvaluator instance from a serialized state."""
        return cls(
            model_name=state.get('model_name'),
            api_key=state.get('api_key')
        )


class AccuracyReward(BaseRewardFunction):
    """
    Reward function that evaluates mathematical correctness.
    
    Uses math_verify as the primary evaluator, with optional LLM judge fallback
    when math_verify fails or returns 0.
    """
    
    def __init__(self, llm_judge: Optional[LLMJudgeEvaluator] = None):
        """
        Initialize the AccuracyReward function.
        
        Args:
            llm_judge: Optional LLM judge evaluator to use as fallback when
                      math_verify fails or returns 0. If None, no fallback is used.
        """
        super().__init__()
        
        # Initialize math_verify (always available)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        
        # Optional LLM judge fallback
        self.llm_judge = llm_judge
    
    def _evaluate_with_math_verify(self, completion_answer: str, ground_truth: str) -> float:
        """Evaluate using math_verify for mathematical equivalence."""
        try:
            completion_boxed = "\\boxed{" + completion_answer + "}"
            ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
            
            # Quick string comparison first
            if ground_truth_boxed.strip().lower() == completion_boxed.strip().lower():
                return 1.0
            
            # Use math_verify for more sophisticated comparison
            score, _ = self.verify_func([ground_truth_boxed], [completion_boxed])
            return float(score)
        except Exception:
            return 0.0
    
    def calculate_rewards(self, completions: List[str], target: Any, 
                         num_generations: int = 1, **kwargs) -> List[float]:
        """
        Calculate accuracy rewards for a list of completions.
        
        Uses math_verify first, then falls back to LLM judge if math_verify
        fails or returns 0 (and LLM judge is configured).
        
        Args:
            completions: List of text completions to evaluate
            target: Ground truth answers (can be list or single value)
            num_generations: Number of generations per prompt
            **kwargs: Additional arguments (unused but maintained for compatibility)
            
        Returns:
            List of float reward values (0.0 to 1.0) for each completion
        """
        rewards = []
        G = max(1, int(num_generations))
        total = len(completions)
        
        for i, completion in enumerate(completions):
            try:
                # Get ground truth for this completion
                ground_truth = _select_for_index(target, i, G, total)
                
                # Extract the answer from the completion
                completion_answer = _extract_boxed_answer(completion)
                
                # First, try math_verify
                reward = self._evaluate_with_math_verify(completion_answer, ground_truth)
                
                # If math_verify failed or returned 0, and we have LLM judge, try it
                if reward == 0.0 and self.llm_judge is not None:
                    try:
                        llm_reward = self.llm_judge.evaluate(completion_answer, ground_truth)
                        # Use LLM judge result if it's better than 0
                        if llm_reward > 0.0:
                            reward = llm_reward
                    except Exception:
                        # If LLM judge fails, stick with math_verify result
                        reward = 0.0
                        logger.warning("LLM judge failed, using math_verify result")
                
            except Exception:
                reward = 0.0
                
            rewards.append(reward)
        
        return rewards
    
    @classmethod
    def with_math_verify_only(cls) -> 'AccuracyReward':
        """Create an AccuracyReward instance using only math_verify."""
        return cls(llm_judge=None)
    
    @classmethod
    def with_llm_fallback(cls, model_name: str, api_key: Optional[str] = None) -> 'AccuracyReward':
        """
        Create an AccuracyReward instance with LLM judge fallback.
        
        Args:
            model_name: Name of the LLM to use for evaluation
            api_key: API key for the LLM service
            
        Returns:
            AccuracyReward instance configured with LLM judge fallback
        """
        llm_judge = LLMJudgeEvaluator(model_name, api_key)
        return cls(llm_judge=llm_judge)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the serializable state of the AccuracyReward function."""
        state = super().get_state()
        state.update({
            'llm_judge_state': self.llm_judge.get_state() if self.llm_judge else None,
        })
        return state
    
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'AccuracyReward':
        """Create an AccuracyReward instance from a serialized state."""
        llm_judge = None
        if state.get('llm_judge_state'):
            llm_judge = LLMJudgeEvaluator.from_state(state['llm_judge_state'])
        
        return cls(llm_judge=llm_judge)
    
    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        # For pickle, we need to handle the math_verify function specially
        # since it may not be directly serializable
        state = self.get_state()
        # Store the fact that we need to reinitialize math_verify
        state['_needs_math_verify_init'] = True
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        # Reconstruct from state
        reconstructed = self.from_state(state)
        self.__dict__.update(reconstructed.__dict__)
        
        # Reinitialize math_verify if needed
        if state.get('_needs_math_verify_init', False):
            self.verify_func = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
