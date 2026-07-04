# SafeDICE Value Function Analysis

This module provides tools to load a trained SafeDICE policy's value function (critic network) and use it to classify and analyze states and trajectories.

## Overview

The scripts analyze the value function learned by SafeDICE during training in two main ways:

1. **State-Level Analysis**: Evaluates individual states to classify them as high/low value 
2. **Trajectory-Level Analysis**: Analyzes entire trajectories to assess their quality and safety

The value function serves as a proxy for:
- **Trajectory quality**: Higher value → better trajectory
- **Safety level**: Higher value → safer trajectory (follows learned safe policy)
- **Reward potential**: Value correlates with achievable rewards

## Files

- `analyze_value_function.py`: Main analysis class and command-line interface
- `example_value_function_usage.py`: Practical examples of using the analyzer
- `README.md`: This documentation

## Installation

No additional dependencies beyond the SafeDICE environment:
```bash
# Ensure SafeDICE environment is set up
cd SafeDICE
conda env create -f environment.yml
conda activate safedice
```

## Quick Start

### Method 1: Command-line Usage

```bash
# Basic analysis with defaults
python analyze_value_function.py

# Custom weights file and dataset
python analyze_value_function.py \
    --weights SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle \
    --dataset SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle \
    --max-trajectories 50 \
    --output-dir value_analysis_results
```

### Method 2: Python API Usage

```python
from analyze_value_function import ValueFunctionAnalyzer

# Initialize analyzer
analyzer = ValueFunctionAnalyzer(
    weights_path='SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
)

# Get value estimates for states
states = ... # shape: (N, state_dim)
values = analyzer.get_state_values(states)

# Classify states as high/low value
values, classifications = analyzer.classify_states_by_value(
    states, 
    method='percentile',
    threshold=0.5  # Classify as high if above median
)

# Analyze a trajectory
trajectory = {
    'observations': trajectory_states,
    'rewards': trajectory_rewards,
    'costs': trajectory_costs
}
result = analyzer.analyze_trajectory(trajectory)

# Analyze full dataset
analysis = analyzer.analyze_dataset('SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle')
```

## API Reference

### ValueFunctionAnalyzer

#### Initialization
```python
analyzer = ValueFunctionAnalyzer(weights_path, config_dict=None)
```

**Parameters:**
- `weights_path` (str): Path to the .pickle weights file from training
- `config_dict` (dict, optional): Configuration dictionary. Defaults to safedice_config

#### get_state_values(states)
Get value function estimates for states.

**Parameters:**
- `states` (np.ndarray): Array of shape (N, state_dim)

**Returns:**
- `values` (np.ndarray): Array of shape (N,) with value estimates

**Example:**
```python
states = np.random.randn(100, 60)
values = analyzer.get_state_values(states)
# values.shape = (100,)
```

#### classify_states_by_value(states, method='percentile', threshold=0.5)
Classify states into high/low value categories.

**Parameters:**
- `states` (np.ndarray): Array of shape (N, state_dim)
- `method` (str): 'percentile' or 'absolute'
  - 'percentile': classify based on percentile of value distribution
  - 'absolute': classify based on absolute threshold value
- `threshold` (float): Classification threshold
  - For 'percentile': value between 0-1 (0.5 = median)
  - For 'absolute': absolute value threshold

**Returns:**
- `values` (np.ndarray): Array of shape (N,) with value estimates
- `classifications` (np.ndarray): Array of shape (N,) with binary classification (0 or 1)

**Example:**
```python
values, classifications = analyzer.classify_states_by_value(
    states,
    method='percentile',
    threshold=0.5  # High value = above median
)
num_high = np.sum(classifications)
num_low = len(classifications) - num_high
```

#### analyze_trajectory(trajectory)
Analyze a single trajectory and compute statistics.

**Parameters:**
- `trajectory` (dict): Dictionary with keys:
  - 'observations': Array of shape (T, state_dim) - required
  - 'rewards': Array of shape (T,) - optional
  - 'costs': Array of shape (T,) - optional

**Returns:**
- `result` (dict): Contains:
  - `trajectory_length`: Length T
  - `value_min`, `value_max`, `value_mean`, `value_std`: Value function statistics
  - `reward_*`, `cost_*`: Statistics for rewards/costs if provided
  - `value_reward_correlation`: Correlation between value and reward (if rewards provided)
  - `value_cost_correlation`: Correlation between value and cost (if costs provided)
  - `estimated_trajectory_quality`: Average value (quality proxy)
  - `values`: Array of individual state values

**Example:**
```python
trajectory = {
    'observations': states,
    'rewards': rewards,
    'costs': costs
}
result = analyzer.analyze_trajectory(trajectory)
print(f"Value mean: {result['value_mean']:.4f}")
print(f"Value-reward correlation: {result['value_reward_correlation']:.4f}")
```

#### analyze_dataset(dataset_path, max_trajectories=None)
Analyze a full dataset of trajectories.

**Parameters:**
- `dataset_path` (str): Path to pickle file containing trajectories
- `max_trajectories` (int, optional): Max number of trajectories to analyze

**Returns:**
- `aggregate_stats` (dict): Contains:
  - `num_trajectories`: Number analyzed
  - `avg_trajectory_length`: Average length
  - `value_statistics`: Global value function stats
  - `reward_statistics`: Global reward stats (if available)
  - `cost_statistics`: Global cost stats (if available)
  - `value_reward_correlation_stats`: Correlation statistics (if available)
  - `detailed_results`: List of individual trajectory analyses

**Example:**
```python
analysis = analyzer.analyze_dataset(
    'SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle',
    max_trajectories=100
)
print(f"Analyzed {analysis['num_trajectories']} trajectories")
print(f"Mean value: {analysis['value_statistics']['mean_of_means']:.4f}")
```

#### plot_value_distribution(states, labels=None, save_path=None)
Plot distribution of value function estimates.

**Parameters:**
- `states` (np.ndarray): Array of shape (N, state_dim)
- `labels` (np.ndarray, optional): Labels for coloring
- `save_path` (str, optional): Path to save figure

**Returns:**
- `fig`: Matplotlib figure

#### plot_trajectory_analysis(trajectory, save_path=None)
Plot value function along with rewards/costs for a trajectory.

**Parameters:**
- `trajectory` (dict): Dictionary with 'observations' and optional 'rewards', 'costs'
- `save_path` (str, optional): Path to save figure

**Returns:**
- `fig`: Matplotlib figure

## Use Cases

### 1. State Classification (High Reward vs Low Reward)

Classify states based on their value:

```python
analyzer = ValueFunctionAnalyzer('weights.pickle')

# Get all states from trajectories
all_states = ... # shape: (N, state_dim)

# Classify as high/low value
values, classifications = analyzer.classify_states_by_value(
    all_states,
    method='percentile',
    threshold=0.75  # Top quartile = high value
)

# Separate states
high_value_states = all_states[classifications == 1]
low_value_states = all_states[classifications == 0]

print(f"High value states: {len(high_value_states)}")
print(f"Low value states: {len(low_value_states)}")
```

### 2. Safe vs Unsafe State Classification

Use value as a proxy for safety (assuming value correlates with following safe policy):

```python
# Classify states
values, safe_mask = analyzer.classify_states_by_value(
    all_states,
    method='percentile',
    threshold=0.33  # Bottom 33% = unsafe
)

safe_states = all_states[safe_mask]
unsafe_states = all_states[~safe_mask]

# If ground truth costs are available, validate
if costs is not None:
    safe_costs = costs[safe_mask]
    unsafe_costs = costs[~safe_mask]
    
    print(f"Safe state mean cost: {np.mean(safe_costs):.4f}")
    print(f"Unsafe state mean cost: {np.mean(unsafe_costs):.4f}")
```

### 3. Trajectory Quality Assessment

Evaluate entire trajectories based on value function:

```python
analyzer = ValueFunctionAnalyzer('weights.pickle')

# Analyze trajectories
analysis = analyzer.analyze_dataset('dataset.pickle', max_trajectories=1000)

# Get trajectories sorted by quality
trajectories = analysis['detailed_results']
sorted_trajs = sorted(trajectories, key=lambda x: x['value_mean'])

# High quality trajectories
print("Top 10 highest quality trajectories:")
for i, traj in enumerate(sorted_trajs[-10:]):
    print(f"  {i}: Value mean={traj['value_mean']:.4f}, Length={traj['trajectory_length']}")

# Low quality trajectories
print("\nBottom 10 lowest quality trajectories:")
for i, traj in enumerate(sorted_trajs[:10]):
    print(f"  {i}: Value mean={traj['value_mean']:.4f}, Length={traj['trajectory_length']}")
```

### 4. Value-Reward Correlation Analysis

Understand relationship between value function and actual rewards:

```python
analysis = analyzer.analyze_dataset('dataset.pickle')

# Check correlation statistics
corr_stats = analysis['value_reward_correlation_stats']
print(f"Value-Reward correlation:")
print(f"  Mean: {corr_stats['mean']:.4f}")
print(f"  Std: {corr_stats['std']:.4f}")

# Strong positive correlation suggests value function learned reward structure well
if corr_stats['mean'] > 0.7:
    print("✅ Strong value-reward correlation - value function is predictive")
elif corr_stats['mean'] > 0.3:
    print("⚠️  Moderate value-reward correlation")
else:
    print("❌ Weak or negative correlation - value function may not predict rewards well")
```

## Interpretation Guide

### Value Function Outputs

- **High values**: States that align well with the learned safe policy
- **Low values**: States that deviate from the learned policy
- **Mean value across trajectory**: Trajectory quality indicator

### Classification Thresholds

**Percentile Method:**
- `threshold=0.5`: Classify above/below median
- `threshold=0.25`: Classify bottom quartile vs rest
- `threshold=0.75`: Classify top quartile vs rest

**Absolute Method:**
- Set threshold to a specific value (requires understanding the value function scale)

### Correlation Interpretation

- **value_reward_correlation > 0.7**: Strong positive relationship
- **value_reward_correlation 0.3-0.7**: Moderate relationship
- **value_reward_correlation < 0.3**: Weak relationship
- **value_reward_correlation < 0**: Negative relationship (unusual)

## Output Examples

### Command-line Output

```
================================================================================
SafeDICE Value Function Analyzer
================================================================================

Loading weights from: SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_...

✅ Model loaded and weights set!

================================================================================
Dataset Analysis
================================================================================

Analyzing 50 trajectories...
100%|██████████| 50/50 [00:12<00:00,  4.15 it/s]

================================================================================
Summary Statistics
================================================================================
Number of trajectories analyzed: 50
Average trajectory length: 847.3

Value Function Statistics:
  global_min: -5.2341
  global_max: 3.8912
  mean_of_means: -0.4523
  std_of_stds: 1.2341

Reward Statistics:
  mean_return: 324.5671
  std_return: 125.4321
  min_return: 12.3456
  max_return: 589.1234

Cost Statistics:
  mean_cost: 2.3456
  std_cost: 1.8901
  min_cost: 0.0000
  max_cost: 7.5432

Value-Reward Correlation:
  Mean correlation: 0.6234
  Std correlation: 0.1892

✅ Saved detailed analysis to: analysis_results.pkl
✅ Saved summary to: analysis_summary.txt
```

### Python API Output

```python
>>> values = analyzer.get_state_values(states)
>>> print(f"Value range: [{values.min():.4f}, {values.max():.4f}]")
Value range: [-5.2341, 3.8912]

>>> values, classifications = analyzer.classify_states_by_value(states, threshold=0.5)
>>> print(f"High value states: {np.sum(classifications)}")
High value states: 541

>>> result = analyzer.analyze_trajectory(trajectory)
>>> print(result['value_reward_correlation'])
0.6234
```

## Troubleshooting

### Error: "Weights file not found"
- Ensure the path is correct relative to where you're running the script
- Use absolute paths if running from a different directory

### Error: "No critic parameters found in checkpoint"
- Ensure the pickle file is from a SafeDICE training run
- Check that the file wasn't corrupted

### Error: "Could not infer state dimension"
- The weights file may have an unexpected structure
- Try manually specifying dimensions

### Value function returns NaN
- May indicate numerical instability
- Try normalizing input states before evaluation

### Poor value-reward correlation
- The value function may not have converged
- Try analyzing different checkpoints from training
- Verify that the training set includes diverse trajectories

## Advanced Usage

### Custom Configuration

```python
custom_config = {
    'hidden_size': 256,
    'gamma': 0.99,
    'grad_reg_coeffs': (10, 1e-6),
    # ... other parameters
}

analyzer = ValueFunctionAnalyzer('weights.pickle', config_dict=custom_config)
```

### Batch Processing

```python
import os
from pathlib import Path

weights_dir = 'SafeDICE/weights_LR_HC'

for weights_file in Path(weights_dir).glob('*.pickle'):
    print(f"Analyzing {weights_file.name}...")
    
    analyzer = ValueFunctionAnalyzer(str(weights_file))
    analysis = analyzer.analyze_dataset('dataset.pickle', max_trajectories=10)
    
    # Save results
    output_file = f"analysis_{weights_file.stem}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(analysis, f)
```

### Real-time State Classification

```python
# For continuous monitoring during rollouts
def classify_current_state(state, analyzer, threshold=0.5):
    state = state.reshape(1, -1)
    value = analyzer.get_state_values(state)[0]
    is_safe = value >= threshold
    return is_safe, value

# In rollout loop:
for t in range(episode_length):
    state = env.get_state()
    is_safe, value = classify_current_state(state, analyzer)
    
    if not is_safe:
        print(f"Warning: Potentially unsafe state detected (value={value:.4f})")
```

## References

For more information on SafeDICE and value function learning:
- SafeDICE paper and documentation in `SafeDICE/README.md`
- Training logs in `SafeDICE/logs_LR_HC/`
- Evaluation results in `SafeDICE/weights_LR_HC/`
