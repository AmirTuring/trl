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


def get_rooted_tree_canonical_form(G: nx.Graph, root: int) -> tuple:
    """
    Compute a canonical form for a tree rooted at a specific node.
    
    This creates a tuple representation of the tree structure that is
    invariant to node relabeling (except for the root which is fixed).
    Two trees rooted at their respective roots are isomorphic iff their
    canonical forms are equal.
    
    Args:
        G: NetworkX graph representing the tree
        root: The root node
        
    Returns:
        A nested tuple representing the canonical form of the rooted tree
    """
    if root not in G:
        return None
    
    def get_subtree_canonical(node: int, parent: int) -> tuple:
        """Recursively compute canonical form of subtree."""
        children = [n for n in G.neighbors(node) if n != parent]
        if not children:
            return ()
        # Get canonical forms of all child subtrees and sort them
        child_forms = sorted(get_subtree_canonical(c, node) for c in children)
        return tuple(child_forms)
    
    return get_subtree_canonical(root, -1)


def are_trees_isomorphic(edges1: List[Tuple[int, int]], 
                         edges2: List[Tuple[int, int]],
                         root_must_match: bool = True) -> bool:
    """
    Check if two trees are isomorphic with node 0 as a fixed root.
    
    Two trees are considered isomorphic if:
    1. They have the same structure (standard graph isomorphism)
    2. Node 0 has the same role in both trees (rooted isomorphism)
    
    This means the trees must look identical when viewed from node 0 as the root,
    even though other nodes may be relabeled.
    
    Args:
        edges1: First tree as list of (node1, node2) tuples
        edges2: Second tree as list of (node1, node2) tuples
        root_must_match: If True, requires node 0 to have the same position
                        in both trees (default True)
        
    Returns:
        True if trees are isomorphic with matching roots, False otherwise
        
    Examples:
        >>> # Same structure, node 0 has same role (degree 2, same subtree structure)
        >>> edges1 = [(0, 1), (0, 2), (1, 3)]
        >>> edges2 = [(0, 2), (0, 1), (2, 3)]
        >>> are_trees_isomorphic(edges1, edges2)
        True
        
        >>> # Same structure but node 0 has different role
        >>> edges1 = [(0, 1), (1, 2), (1, 3)]  # node 0 is a leaf
        >>> edges2 = [(0, 1), (0, 2), (2, 3)]  # node 0 has degree 2
        >>> are_trees_isomorphic(edges1, edges2)
        False
        
        >>> # Different structure
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
    
    # First check basic isomorphism
    if not nx.is_isomorphic(G1, G2):
        return False
    
    if not root_must_match:
        return True
    
    # Check that node 0 exists in both graphs
    if 0 not in G1.nodes or 0 not in G2.nodes:
        # If neither has node 0, they could still be isomorphic
        if 0 not in G1.nodes and 0 not in G2.nodes:
            return True
        return False
    
    # Check rooted isomorphism: node 0 must have the same structural role
    # Compare canonical forms of trees rooted at node 0
    canonical1 = get_rooted_tree_canonical_form(G1, 0)
    canonical2 = get_rooted_tree_canonical_form(G2, 0)
    
    return canonical1 == canonical2


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
    Reward function that checks if the predicted number of shapes is correct.
    
    Args:
        completions: List of model completions (each is a list with one dict containing 'content')
        num_nodes: List of ground truth number of shapes
        **kwargs: Additional keyword arguments
        
    Returns:
        List of float rewards (1.0 for correct, 0.0 for wrong)
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
                rewards.append(0.0)
                
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
