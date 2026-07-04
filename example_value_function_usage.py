#!/usr/bin/env python
"""
Example usage script for the Value Function Analyzer.
Demonstrates various ways to use the value function for classification.
"""

import numpy as np
import tensorflow as tf
import pickle
import sys
from pathlib import Path

# Add SafeDICE to path
safedice_path = Path(__file__).parent / 'SafeDICE'
sys.path.insert(0, str(safedice_path))

from analyze_value_function import ValueFunctionAnalyzer


def example_1_basic_value_estimation():
    """Example 1: Basic value estimation for random states."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Value Estimation")
    print("=" * 80)
    
    weights_path = 'SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
    
    # Initialize analyzer
    analyzer = ValueFunctionAnalyzer(weights_path)
    
    # Generate random states (assuming PointGoal1 environment with ~60 state dims)
    random_states = np.random.randn(10, 60).astype(np.float32)
    
    # Get value estimates
    values = analyzer.get_state_values(random_states)
    
    print(f"\nRandom states shape: {random_states.shape}")
    print(f"Value estimates shape: {values.shape}")
    print(f"Value statistics:")
    print(f"  Min: {np.min(values):.4f}")
    print(f"  Max: {np.max(values):.4f}")
    print(f"  Mean: {np.mean(values):.4f}")
    print(f"  Std: {np.std(values):.4f}")
    
    return analyzer, values


def example_2_state_classification():
    """Example 2: Classify states as high/low value."""
    print("\n" + "=" * 80)
    print("Example 2: State Classification (High/Low Value)")
    print("=" * 80)
    
    weights_path = 'SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
    
    analyzer = ValueFunctionAnalyzer(weights_path)
    
    # Generate random states
    random_states = np.random.randn(100, 60).astype(np.float32)
    
    # Classify using percentile method (50th percentile = median)
    values, classifications = analyzer.classify_states_by_value(
        random_states, 
        method='percentile',
        threshold=0.5  # Classify as high if above median
    )
    
    num_high_value = np.sum(classifications)
    num_low_value = len(classifications) - num_high_value
    
    print(f"\nClassification Results (Percentile Method):")
    print(f"  High value states: {num_high_value}")
    print(f"  Low value states: {num_low_value}")
    print(f"  Percentage high: {100 * num_high_value / len(classifications):.1f}%")
    
    # Also try quartile division
    values, q1_classification = analyzer.classify_states_by_value(
        random_states,
        method='percentile',
        threshold=0.25  # Bottom 25%
    )
    
    values, q3_classification = analyzer.classify_states_by_value(
        random_states,
        method='percentile',
        threshold=0.75  # Top 25%
    )
    
    print(f"\nQuartile Division:")
    print(f"  Bottom 25% (low value): {np.sum(1 - q1_classification)}")
    print(f"  Top 25% (high value): {np.sum(q3_classification)}")
    print(f"  Middle 50%: {100 - np.sum(1 - q1_classification) - np.sum(q3_classification)}")


def example_3_trajectory_analysis():
    """Example 3: Analyze a single trajectory."""
    print("\n" + "=" * 80)
    print("Example 3: Trajectory Analysis")
    print("=" * 80)
    
    weights_path = 'SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
    
    analyzer = ValueFunctionAnalyzer(weights_path)
    
    # Load dataset
    dataset_path = 'SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle'
    print(f"\nLoading dataset from: {dataset_path}")
    
    try:
        import pickle5 as pickle_lib
    except ImportError:
        import pickle as pickle_lib
    
    with open(dataset_path, 'rb') as f:
        data = pickle_lib.load(f)
    
    # Get first trajectory
    if isinstance(data, dict) and 'observations' in data:
        observations = data['observations'][:1000]  # First 1000 steps
        rewards = data.get('rewards', np.zeros(len(observations)))[:1000]
        costs = data.get('costs', np.zeros(len(observations)))[:1000]
        
        trajectory = {
            'observations': observations,
            'rewards': rewards,
            'costs': costs,
        }
    else:
        print("Could not extract trajectory from dataset")
        return
    
    # Analyze trajectory
    result = analyzer.analyze_trajectory(trajectory)
    
    print(f"\nTrajectory Analysis Results:")
    print(f"  Length: {result['trajectory_length']}")
    print(f"  Value function:")
    print(f"    Min: {result['value_min']:.4f}")
    print(f"    Max: {result['value_max']:.4f}")
    print(f"    Mean: {result['value_mean']:.4f}")
    print(f"    Std: {result['value_std']:.4f}")
    
    if 'reward_mean' in result:
        print(f"  Rewards:")
        print(f"    Mean: {result['reward_mean']:.4f}")
        print(f"    Total: {result['reward_traj_return']:.4f}")
        print(f"    Value-Reward Correlation: {result['value_reward_correlation']:.4f}")
    
    if 'cost_mean' in result:
        print(f"  Costs:")
        print(f"    Mean: {result['cost_mean']:.4f}")
        print(f"    Total: {result['cost_accumulation']:.4f}")
        print(f"    Value-Cost Correlation: {result['value_cost_correlation']:.4f}")


def example_4_dataset_statistical_analysis():
    """Example 4: Statistical analysis of value function across dataset."""
    print("\n" + "=" * 80)
    print("Example 4: Dataset Statistical Analysis")
    print("=" * 80)
    
    weights_path = 'SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
    dataset_path = 'SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle'
    
    analyzer = ValueFunctionAnalyzer(weights_path)
    
    print(f"\nAnalyzing dataset...")
    analysis = analyzer.analyze_dataset(dataset_path, max_trajectories=50)
    
    print(f"\n✅ Analysis Complete!")
    print(f"\nDataset Summary:")
    print(f"  Number of trajectories: {analysis['num_trajectories']}")
    print(f"  Average trajectory length: {analysis['avg_trajectory_length']:.1f}")
    
    print(f"\nValue Function Statistics:")
    for key, val in analysis['value_statistics'].items():
        print(f"  {key}: {val:.6f}")
    
    if 'reward_statistics' in analysis:
        print(f"\nReward Statistics:")
        for key, val in analysis['reward_statistics'].items():
            print(f"  {key}: {val:.6f}")
    
    if 'cost_statistics' in analysis:
        print(f"\nCost Statistics:")
        for key, val in analysis['cost_statistics'].items():
            print(f"  {key}: {val:.6f}")
    
    if 'value_reward_correlation_stats' in analysis:
        print(f"\nValue-Reward Correlation Analysis:")
        corr_stats = analysis['value_reward_correlation_stats']
        print(f"  Mean: {corr_stats['mean']:.6f}")
        print(f"  Std: {corr_stats['std']:.6f}")
        print(f"  Individual correlations (first 5): {corr_stats['correlations'][:5]}")


def example_5_safe_unsafe_classification():
    """
    Example 5: Classify trajectories as safe/unsafe based on value function.
    
    Here, we use the value function as a proxy for trajectory safety:
    - High value ≈ trajectory follows learned safe policy (should be safe)
    - Low value ≈ trajectory deviates from learned policy (potentially unsafe)
    """
    print("\n" + "=" * 80)
    print("Example 5: Safe/Unsafe Trajectory Classification")
    print("=" * 80)
    
    weights_path = 'SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
    dataset_path = 'SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle'
    
    analyzer = ValueFunctionAnalyzer(weights_path)
    
    try:
        import pickle5 as pickle_lib
    except ImportError:
        import pickle as pickle_lib
    
    with open(dataset_path, 'rb') as f:
        data = pickle_lib.load(f)
    
    # Get all states
    if isinstance(data, dict) and 'observations' in data:
        all_states = data['observations']
        all_costs = data.get('costs', np.zeros(len(all_states)))
    else:
        print("Could not extract data from dataset")
        return
    
    # Get values
    values = analyzer.get_state_values(all_states)
    
    # Method 1: Classification based on value percentiles
    print("\nMethod 1: Value-based Classification")
    value_threshold = np.percentile(values, 33)  # Bottom 33% = unsafe
    safe_mask = values >= value_threshold
    unsafe_mask = ~safe_mask
    
    print(f"  Safe states (high value): {np.sum(safe_mask)} ({100*np.sum(safe_mask)/len(safe_mask):.1f}%)")
    print(f"  Unsafe states (low value): {np.sum(unsafe_mask)} ({100*np.sum(unsafe_mask)/len(safe_mask):.1f}%)")
    
    # Compare with actual costs
    if all_costs is not None:
        safe_state_costs = all_costs[safe_mask]
        unsafe_state_costs = all_costs[unsafe_mask]
        
        print(f"\n  Actual costs comparison:")
        print(f"    Safe states - Mean cost: {np.mean(safe_state_costs):.6f}")
        print(f"    Unsafe states - Mean cost: {np.mean(unsafe_state_costs):.6f}")
        print(f"    Cost ratio (unsafe/safe): {np.mean(unsafe_state_costs) / np.mean(safe_state_costs):.2f}x")
    
    # Method 2: Trajectory-level classification
    print(f"\nMethod 2: Trajectory-level Safety Classification")
    
    trajectory_values = []
    trajectory_costs = []
    
    # Simple episode segmentation (assuming episodes are <= 1000 steps)
    episode_length = 1000
    for i in range(0, len(all_states), episode_length):
        episode_vals = values[i:i+episode_length]
        if len(episode_vals) > 0:
            trajectory_values.append(np.mean(episode_vals))
            trajectory_costs.append(np.sum(all_costs[i:i+episode_length]))
    
    trajectory_values = np.array(trajectory_values)
    trajectory_costs = np.array(trajectory_costs)
    
    # Classify trajectories
    trajectory_threshold = np.median(trajectory_values)
    safe_trajectories = trajectory_values >= trajectory_threshold
    
    print(f"  Total trajectories: {len(trajectory_values)}")
    print(f"  Safe trajectories (high value): {np.sum(safe_trajectories)}")
    print(f"  Unsafe trajectories (low value): {np.sum(~safe_trajectories)}")
    
    if len(trajectory_costs) > 0:
        safe_traj_costs = trajectory_costs[safe_trajectories]
        unsafe_traj_costs = trajectory_costs[~safe_trajectories]
        
        print(f"\n  Cumulative costs:")
        print(f"    Safe trajectories - Mean: {np.mean(safe_traj_costs):.4f}, Std: {np.std(safe_traj_costs):.4f}")
        print(f"    Unsafe trajectories - Mean: {np.mean(unsafe_traj_costs):.4f}, Std: {np.std(unsafe_traj_costs):.4f}")
        print(f"    Cost ratio (unsafe/safe): {np.mean(unsafe_traj_costs) / np.mean(safe_traj_costs):.2f}x")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("SafeDICE Value Function Analysis - Example Usage")
    print("=" * 80)
    
    try:
        # Run examples
        example_1_basic_value_estimation()
        example_2_state_classification()
        example_3_trajectory_analysis()
        example_4_dataset_statistical_analysis()
        example_5_safe_unsafe_classification()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
