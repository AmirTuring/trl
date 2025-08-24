#!/usr/bin/env python3
"""
Demonstration of Enhanced GRPO Reward Logging

This script demonstrates the new reward logging capabilities without running full training.
It shows how the reward functions work and what kind of detailed logging they provide.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from grpo_math import format_reward_func, accuracy_reward, REWARD_STORAGE, get_problem_reward_stats, print_reward_summary

# Sample completions for testing
sample_completions = [
    "<think>I need to solve this step by step. First, I'll identify what we're looking for.</think> The answer is \\boxed{42}",
    "Let me think about this problem. We have x + 5 = 10, so x = 5. Therefore, the answer is \\boxed{5}",
    "<think>This is a simple arithmetic problem. 2 + 2 = 4. The answer is \\boxed{4}</think>",
    "I need to be careful here. Let me work through this systematically. The final answer is \\boxed{100}",
    "Quick calculation: 3 * 7 = 21. So the answer is \\boxed{21}",
]

sample_targets = ["42", "5", "4", "100", "21"]

sample_prompts = [
    "What is 6 * 7?",
    "Solve for x: x + 5 = 10",
    "What is 2 + 2?", 
    "What is 10^2?",
    "What is 3 * 7?",
]

def demo_format_rewards():
    """Demonstrate format reward function."""
    print("=== FORMAT REWARD DEMONSTRATION ===")
    print("Testing format reward function with sample completions...\n")
    
    # Test with sample data and dataset indices
    dataset_indices = [0, 1, 2, 3, 4]  # Each completion corresponds to a different problem
    
    rewards = format_reward_func(
        completions=sample_completions,
        target=sample_targets,
        dataset_indices=dataset_indices
    )
    
    print("Format Reward Results:")
    for i, (completion, reward, idx) in enumerate(zip(sample_completions, rewards, dataset_indices)):
        print(f"  Problem {idx}, Completion {i+1}: {reward}")
        print(f"    Text: {completion[:60]}...")
        print()
    
    # Show stored rewards
    print("Stored rewards per problem:")
    for idx in dataset_indices:
        if idx in REWARD_STORAGE['format_reward']:
            print(f"  Problem {idx}: {REWARD_STORAGE['format_reward'][idx]}")

def demo_accuracy_rewards():
    """Demonstrate accuracy reward function."""
    print("=== ACCURACY REWARD DEMONSTRATION ===")
    print("Testing accuracy reward function with sample completions...\n")
    
    try:
        # Test with sample data and dataset indices
        dataset_indices = [0, 1, 2, 3, 4]  # Each completion corresponds to a different problem
        
        rewards = accuracy_reward(
            completions=sample_completions,
            target=sample_targets,
            dataset_indices=dataset_indices
        )
        
        print("Accuracy Reward Results:")
        for i, (completion, target, reward, idx) in enumerate(zip(sample_completions, sample_targets, rewards, dataset_indices)):
            print(f"  Problem {idx}: {reward}")
            print(f"    Expected: {target}")
            print(f"    Completion: {completion[:60]}...")
            print()
        
        # Show stored rewards
        print("Stored rewards per problem:")
        for idx in dataset_indices:
            if idx in REWARD_STORAGE['accuracy_reward']:
                print(f"  Problem {idx}: {REWARD_STORAGE['accuracy_reward'][idx]}")
                
    except Exception as e:
        print(f"Error running accuracy reward (math_verify may not be installed): {e}")
        # Simulate some results for demo
        print("Simulating accuracy rewards for demonstration...")
        dataset_indices = [0, 1, 2, 3, 4]
        for idx in dataset_indices:
            if idx not in REWARD_STORAGE['accuracy_reward']:
                REWARD_STORAGE['accuracy_reward'][idx] = [1.0]  # Simulate correct answer

def demo_logging_structure():
    """Show the structure of logged data."""
    print("=== LOGGING STRUCTURE DEMONSTRATION ===")
    print("The enhanced GRPO trainer logs the following information:\n")
    
    print("1. Problem-based Reward Storage:")
    print("   - REWARD_STORAGE dictionary with dataset indices as keys")
    print("   - Separate lists for each reward function")
    print("   - Accumulated rewards per problem across training")
    print()
    
    print("2. Reward Statistics Files (reward_statistics/):")
    print("   - JSON files with per-problem statistics")
    print("   - Mean, std, min, max, count for each problem")
    print("   - Timestamped files for tracking progress")
    print()
    
    print("3. Console Output During Training:")
    print("   - Periodic logging every 50 steps")
    print("   - Real-time feedback per problem")
    print("   - Format: 'Problem X: mean ± std (count evals)'")
    print()
    
    print("4. Final Training Summary:")
    print("   - Overall statistics across all problems")
    print("   - Top/bottom performing problems")
    print("   - Problem difficulty analysis")
    print()
    
    print("5. Analysis Tools:")
    print("   - analyze_problem_rewards.py for detailed analysis")
    print("   - demo_problem_tracking.py for testing")
    print("   - JSON output for custom analysis")

def main():
    print("Enhanced GRPO Reward Logging Demonstration")
    print("=" * 50)
    print()
    
    # Clean up any existing logs for demo
    import shutil
    for dir_name in ["reward_statistics", "completion_samples"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    # Clear REWARD_STORAGE for clean demo
    REWARD_STORAGE['format_reward'].clear()
    REWARD_STORAGE['accuracy_reward'].clear()
    
    # Run demonstrations
    demo_format_rewards()
    print()
    demo_accuracy_rewards()
    print()
    demo_logging_structure()
    
    print("\n" + "=" * 50)
    print("Demo complete! Check the generated files:")
    
    # Show generated files
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(keyword in file for keyword in ["reward", "completion", "statistics"]):
                full_path = os.path.join(root, file)
                print(f"  {full_path}")
    
    # Show REWARD_STORAGE contents
    print("\nREWARD_STORAGE contents:")
    for func_name, storage in REWARD_STORAGE.items():
        print(f"  {func_name}: {dict(storage)}")

if __name__ == "__main__":
    main()
