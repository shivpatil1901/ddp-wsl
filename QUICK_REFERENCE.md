# SafeDICE Value Function Analysis - Quick Reference

## TL;DR - Get Started in 2 Minutes

```bash
# Run analysis with defaults
python analyze_value_function.py

# Analyze specific weights file
python analyze_value_function.py --weights SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle

# Run examples
python example_value_function_usage.py

# Filter trajectories
python filter_trajectories_by_value.py --filter-type analyze
```

## What These Scripts Do

### 1. `analyze_value_function.py`
**Main script for value function analysis**

- Loads trained SafeDICE weights
- Extracts and evaluates the critic (value function)
- Classifies states as high/low value
- Analyzes trajectories and datasets
- Generates statistics and visualizations

**Key Features:**
- State-level value estimation
- Trajectory quality assessment
- Value-reward correlation analysis
- Batch dataset processing

**Command-line Usage:**
```bash
python analyze_value_function.py \
    --weights <weights_file.pickle> \
    --dataset <dataset_file.pickle> \
    --max-trajectories 50 \
    --output-dir analysis_results
```

**Python API:**
```python
from analyze_value_function import ValueFunctionAnalyzer

analyzer = ValueFunctionAnalyzer('weights.pickle')
values = analyzer.get_state_values(states)
analysis = analyzer.analyze_dataset('dataset.pickle')
```

---

### 2. `example_value_function_usage.py`
**Practical examples and demonstrations**

Contains 5 complete working examples:
1. **Basic value estimation** - Get values for random states
2. **State classification** - Classify as high/low value
3. **Trajectory analysis** - Analyze single trajectories
4. **Dataset statistics** - Batch process entire datasets
5. **Safe/unsafe classification** - Use value as safety proxy

Run all examples:
```bash
python example_value_function_usage.py
```

Each example shows:
- How to set up the analyzer
- How to call key methods
- How to interpret results
- Print output showing expected results

---

### 3. `filter_trajectories_by_value.py`
**Practical utilities for trajectory filtering and clustering**

**Features:**
- Filter by value thresholds
- Cluster trajectories by quality
- Select top-quality/diverse trajectories
- Create new filtered datasets

**Usage:**
```bash
# Analyze quality distribution
python filter_trajectories_by_value.py --filter-type analyze

# Select top 25% quality trajectories
python filter_trajectories_by_value.py \
    --filter-type high_quality \
    --top-percent 25 \
    --output-dataset filtered_dataset.pickle

# Cluster into 3 groups
python filter_trajectories_by_value.py \
    --filter-type cluster \
    --num-clusters 3 \
    --output-dataset clustered_dataset.pickle
```

---

## Core Concepts

### Value Function
- Learned by the critic network during SafeDICE training
- Takes a state as input, outputs a scalar value
- Represents trajectory quality / safety potential

### Classification Methods

**Percentile-based (Recommended):**
```python
values, labels = analyzer.classify_states_by_value(
    states,
    method='percentile',
    threshold=0.5  # Above median
)
```
- `threshold=0.5`: Split at median
- `threshold=0.25`: Bottom quartile
- `threshold=0.75`: Top quartile

**Absolute-based:**
```python
values, labels = analyzer.classify_states_by_value(
    states,
    method='absolute',
    threshold=-1.0  # Value >= -1.0
)
```

### Trajectory Analysis
Returns dictionary with:
- `value_mean`, `value_std`, `value_min`, `value_max`
- `reward_mean`, `reward_sum` (if available)
- `cost_mean`, `cost_sum` (if available)
- `value_reward_correlation`
- `estimated_trajectory_quality`

---

## Common Use Cases & Solutions

### Use Case 1: Identify High-Quality Trajectories
```python
analyzer = ValueFunctionAnalyzer('weights.pickle')

# Analyze dataset
analysis = analyzer.analyze_dataset('dataset.pickle')

# Sort by quality
results = analysis['detailed_results']
results_sorted = sorted(results, key=lambda x: x['value_mean'], reverse=True)

# Top 10 trajectories
for i, traj in enumerate(results_sorted[:10]):
    print(f"{i}: value_mean={traj['value_mean']:.4f}")
```

### Use Case 2: Find Safe vs Unsafe States
```python
# Classify states based on value
values, safe_mask = analyzer.classify_states_by_value(
    all_states,
    method='percentile',
    threshold=0.33  # Bottom third = unsafe
)

# Verify with ground truth if available
safe_states = all_states[safe_mask]
unsafe_states = all_states[~safe_mask]
```

### Use Case 3: Understand Value-Reward Relationship
```python
analysis = analyzer.analyze_dataset('dataset.pickle')

corr_stats = analysis['value_reward_correlation_stats']
print(f"Correlation: {corr_stats['mean']:.4f}")

# Visualize
analyzer.plot_value_distribution(all_states, save_path='value_dist.png')
```

### Use Case 4: Create Filtered Training Dataset
```python
filter_util = TrajectoryFilter(analyzer)

# Select top 25% quality trajectories
metadata = filter_util.create_filtered_dataset(
    'input.pickle',
    'output_high_quality.pickle',
    filter_type='high_quality',
    top_percent=25
)
```

### Use Case 5: Cluster Trajectories by Quality
```python
filter_util = TrajectoryFilter(analyzer)

# Group into 3 quality tiers
clusters = filter_util.cluster_trajectories(
    trajectories,
    num_clusters=3
)

for cluster_id, cluster_data in clusters.items():
    print(f"Cluster {cluster_id}: {cluster_data['num_trajectories']} trajectories")
    print(f"  Mean value: {cluster_data['mean_score']:.4f}")
```

---

## Output Files

When running analysis, generates:

```
value_function_analysis/
├── analysis_results.pkl          # Detailed results (can be large)
├── analysis_summary.txt          # Text summary of statistics
└── *.png                         # Plots (if --plot-distribution used)
```

Load detailed results:
```python
import pickle

with open('value_function_analysis/analysis_results.pkl', 'rb') as f:
    analysis = pickle.load(f)

print(analysis['num_trajectories'])
print(analysis['value_statistics'])
```

---

## Interpretation Guide

### Value Function Statistics

| Metric | What It Means | Good Range |
|--------|---------------|-----------|
| `value_mean` | Average trajectory quality | Depends on distribution |
| `value_std` | Variability within trajectory | - |
| `value_reward_correlation` | How well value predicts reward | > 0.5 is good |
| `mean_cost` (if available) | Average safety cost | Lower is better |

### Classification Accuracy

Check correlation to validate classifications:
- **Correlation > 0.7**: Strong relationship (classify confidently)
- **Correlation 0.3-0.7**: Moderate relationship (use with caution)
- **Correlation < 0.3**: Weak relationship (may not be predictive)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FileNotFoundError: weights.pickle not found" | Use correct path, check file exists |
| "No critic parameters found in checkpoint" | Ensure pickle file is from SafeDICE training |
| "Could not infer state dimension" | Try manually specifying state_dim |
| "Value returns NaN" | Check input states are normalized |
| "Poor correlations" | Try different checkpoint or dataset |

---

## Method Comparison

### State Classification Methods

| Method | Threshold | When to Use | Pros | Cons |
|--------|-----------|-----------|------|------|
| **Percentile** | 0-100% | Most cases | Data-adaptive | Depends on data |
| **Absolute** | Value threshold | Known bounds | Fixed threshold | Requires calibration |
| **Percentile + Quartiles** | 0.25, 0.75 | Fine-grained | More granular | Smaller groups |

### Trajectory Filtering

| Filter Type | Use Case | Example |
|------------|----------|---------|
| **High quality** | Get best trajectories | Top 25% |
| **Low quality** | Find failure cases | Bottom 25% |
| **Cluster** | Understand distribution | 3 quality tiers |
| **Diverse** | Stratified sampling | 100 diverse examples |

---

## Performance Tips

**For Large Datasets:**
```python
# Limit trajectories
analysis = analyzer.analyze_dataset('dataset.pickle', max_trajectories=1000)

# Or batch process
analyzezer = ValueFunctionAnalyzer('weights.pickle')
for i in range(0, num_trajectories, batch_size):
    batch = trajectories[i:i+batch_size]
    results = [analyzer.analyze_trajectory(t) for t in batch]
```

**For GPU Acceleration:**
```python
import tensorflow as tf

# GPU memory growth (default in analyzer)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

---

## Outputs Summary

### Command-line
- Prints statistics to console
- Saves pickle file with detailed results
- Saves text file with summary

### Python API
Returns dictionaries with:
- Individual value estimates
- Trajectory statistics
- Dataset aggregates
- Correlation metrics

### Plots
If enabled:
- Value distribution histogram
- Sorted value function curve
- Trajectory-level value/reward/cost plots

---

## Integration with Other Tools

### With SafeDICE Training
```python
# Extract critic from trained model
from SafeDICE.algorithms.safedice import SafeDICE

model = SafeDICE(state_dim, action_dim, config)
# ... load weights ...
critic = model.critic
```

### With Evaluation Pipeline
```python
# Use value function during rollouts
analyzer = ValueFunctionAnalyzer('weights.pickle')

for state in rollout_trajectory:
    value = analyzer.get_state_values(state.reshape(1, -1))[0]
    confidence = abs(value) / max(abs_value_range)
    # Use confidence in policy decisions
```

### With Data Processing
```python
# Filter datasets for training
filter_util = TrajectoryFilter(analyzer)
high_quality = filter_util.select_high_quality_trajectories(
    trajectories, top_percent=25
)
# Use high_quality for fine-tuning
```

---

## Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `analyze_value_function.py` | Main analyzer class | ~800 |
| `example_value_function_usage.py` | 5 working examples | ~400 |
| `filter_trajectories_by_value.py` | Filtering utilities | ~400 |
| `VALUE_FUNCTION_ANALYSIS_README.md` | Full documentation | ~700 |

---

## Quick API Reference Card

```python
# Initialize
analyzer = ValueFunctionAnalyzer('path/to/weights.pickle')

# Evaluate states
values = analyzer.get_state_values(states)  # (N,)

# Classify states
values, labels = analyzer.classify_states_by_value(
    states, method='percentile', threshold=0.5
)

# Analyze single trajectory
result = analyzer.analyze_trajectory({
    'observations': states,
    'rewards': rewards,
    'costs': costs
})

# Analyze dataset
analysis = analyzer.analyze_dataset('dataset.pickle')

# Plot
analyzer.plot_value_distribution(states, save_path='out.png')
analyzer.plot_trajectory_analysis(trajectory, save_path='out.png')

# Filter
from filter_trajectories_by_value import TrajectoryFilter
filter_util = TrajectoryFilter(analyzer)
filtered = filter_util.select_high_quality_trajectories(trajs, 25)
clusters = filter_util.cluster_trajectories(trajs, 3)
```

---

## Next Steps

1. **Start Simple**: Run `python analyze_value_function.py` with defaults
2. **Explore**: Check `example_value_function_usage.py` for ideas
3. **Customize**: Modify parameters for your specific needs
4. **Integrate**: Use in your pipeline with the API
5. **Analyze**: Create custom analysis for your use case

---

**Questions?** Check `VALUE_FUNCTION_ANALYSIS_README.md` for full documentation.
