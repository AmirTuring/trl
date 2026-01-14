"""
Region Tree Reward Functions for GRPO VLM Training.

This module contains reward functions for training VLMs on:
1. Region counting in images of non-crossing closed curves
2. Region-adjacency tree generation from such images
"""

import re
from typing import Optional, List, Tuple

import networkx as nx

from .base import BaseRewardFunction


def extract_answer_content(content: str) -> Optional[str]:
    """
    Extract content from <answer></answer> tags.
    
    Returns the content inside the tags, or None if not found.
    """
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_tree_from_completion(content: str) -> list[tuple[int, int]]:
    """
    Parse the tree edge list from the model's completion.
    
    Extracts from <answer></answer> tags only.
    
    Expects format like:
    <answer>
    0 1
    1 2
    1 3
    </answer>
    
    Returns a list of tuples representing edges, or empty list if parsing fails.
    """
    edges = []
    
    answer_content = extract_answer_content(content)
    if not answer_content:
        return edges
    
    # Parse edges from the answer content
    lines = answer_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                edges.append((a, b))
            except ValueError:
                continue
    
    return edges


def parse_num_nodes_from_completion(content: str) -> Optional[int]:
    """
    Parse the number of regions from the model's completion.
    
    Extracts from <answer></answer> tags only.
    
    Expects format like:
    <answer>
    5
    </answer>
    
    Returns the number, or None if parsing fails.
    """
    answer_content = extract_answer_content(content)
    if not answer_content:
        return None
    
    # Parse the number from the answer content
    try:
        return int(answer_content.strip())
    except ValueError:
        return None


def are_trees_isomorphic(edges1: List[Tuple[int, int]], 
                         edges2: List[Tuple[int, int]]) -> bool:
    """
    Check if two trees are isomorphic.
    
    Two trees are isomorphic if they have the same structure, meaning
    one can be obtained from the other by relabeling nodes.
    
    Args:
        edges1: First tree as list of (node1, node2) tuples
        edges2: Second tree as list of (node1, node2) tuples
        
    Returns:
        True if trees are isomorphic, False otherwise
        
    Examples:
        >>> edges1 = [(0, 1), (0, 2), (1, 3)]
        >>> edges2 = [(0, 2), (0, 1), (2, 3)]
        >>> are_trees_isomorphic(edges1, edges2)
        True
        
        >>> edges1 = [(0, 1), (0, 2)]
        >>> edges2 = [(0, 1), (1, 2)]
        >>> are_trees_isomorphic(edges1, edges2)
        False
    """
    # Handle empty trees (single node - root only)
    if not edges1 and not edges2:
        return True
    if not edges1 or not edges2:
        return False
    
    # Create undirected graphs from edge lists (trees are undirected)
    G1 = nx.Graph(edges1)
    G2 = nx.Graph(edges2)
    
    # Check if they have the same number of nodes
    if len(G1.nodes) != len(G2.nodes):
        return False
    
    # Check if they have the same number of edges
    if len(G1.edges) != len(G2.edges):
        return False
    
    # Use NetworkX isomorphism checker
    return nx.is_isomorphic(G1, G2)


def tree_correctness_reward(completions, tree: list[list[int]], **kwargs) -> list[float]:
    """
    Reward function that checks if the predicted tree is isomorphic to the ground truth.
    
    Uses tree isomorphism checking - two trees are considered correct if they have
    the same structure, even if node labels differ.
    
    Args:
        completions: List of model completions (each is a list with one dict containing 'content')
        tree: List of ground truth trees (each is a list of edges like [[0,1], [1,2]])
        **kwargs: Additional keyword arguments
        
    Returns:
        List of float rewards (1.0 for isomorphic, 0.0 for not isomorphic)
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    
    for i, (content, gt_tree) in enumerate(zip(contents, tree)):
        try:
            # Parse predicted tree from completion
            pred_edges = parse_tree_from_completion(content)
            
            if not pred_edges and gt_tree:
                # Predicted empty but ground truth has edges
                rewards.append(0.0)
                continue
            
            # Convert ground truth to list of tuples
            gt_edges = [(edge[0], edge[1]) for edge in gt_tree]
            
            # Check if trees are isomorphic
            if are_trees_isomorphic(pred_edges, gt_edges):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                    
        except Exception as e:
            print(f"Tree parsing error: {e}, content: {content[:200]}...")
            rewards.append(0.0)
    
    return rewards


def num_nodes_reward(completions, num_nodes: list[int], **kwargs) -> list[float]:
    """
    Reward function that checks if the predicted number of nodes/regions is correct.
    
    Args:
        completions: List of model completions (each is a list with one dict containing 'content')
        num_nodes: List of ground truth number of nodes
        **kwargs: Additional keyword arguments
        
    Returns:
        List of float rewards (1.0 for correct, partial for close, 0.0 for wrong)
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    
    for content, gt_num in zip(contents, num_nodes):
        try:
            pred_num = parse_num_nodes_from_completion(content)
            
            if pred_num is None:
                rewards.append(0.0)
                continue
            
            if pred_num == gt_num:
                rewards.append(1.0)
            else:
                # Partial reward based on how close the prediction is
                # Using inverse of relative error, capped at 0
                error = abs(pred_num - gt_num) / max(gt_num, 1)
                partial_reward = max(0.0, 1.0 - error)
                rewards.append(partial_reward)
                
        except Exception as e:
            print(f"Num nodes parsing error: {e}, content: {content[:200]}...")
            rewards.append(0.0)
    
    return rewards


class TreeCorrectnessReward(BaseRewardFunction):
    """
    Reward function class for evaluating region-adjacency tree correctness.
    
    Uses tree isomorphism checking - two trees are considered correct if they 
    have the same structure, even if node labels differ.
    """
    
    def __init__(self):
        super().__init__()
        self.__name__ = "TreeCorrectnessReward"
    
    def calculate_rewards(self, completions, tree: list[list[int]] = None, **kwargs) -> list[float]:
        """Calculate tree correctness rewards using isomorphism checking."""
        if tree is None:
            tree = kwargs.get('tree', [])
        return tree_correctness_reward(completions, tree, **kwargs)


class NumNodesReward(BaseRewardFunction):
    """
    Reward function class for evaluating region count correctness.
    
    Compares predicted number of regions with ground truth, providing
    partial credit for close predictions.
    """
    
    def __init__(self):
        super().__init__()
        self.__name__ = "NumNodesReward"
    
    def calculate_rewards(self, completions, num_nodes: list[int] = None, **kwargs) -> list[float]:
        """Calculate num_nodes correctness rewards."""
        if num_nodes is None:
            num_nodes = kwargs.get('num_nodes', [])
        return num_nodes_reward(completions, num_nodes, **kwargs)


def think_answer_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """
    Reward function that checks if the response follows the <think></think> + <answer></answer> format.
    
    Expected format:
    <think>
    [reasoning]
    </think>
    <answer>
    [answer content]
    </answer>
    
    Returns 1.0 if format is correct, 0.0 otherwise.
    """
    # Pattern: <think>...</think> followed by <answer>...</answer>
    # No nested <think> tags allowed, answer must come after think
    pattern = r"^<think>(?!.*<think>)(.*?)</think>\s*<answer>(.*?)</answer>\s*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


class ThinkAnswerFormatReward(BaseRewardFunction):
    """
    Reward function class for validating <think></think> + <answer></answer> format.
    """
    
    def __init__(self):
        super().__init__()
        self.__name__ = "ThinkAnswerFormatReward"
    
    def calculate_rewards(self, completions, **kwargs) -> list[float]:
        """Calculate format rewards."""
        return think_answer_format_reward(completions, **kwargs)
