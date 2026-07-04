# SafeDICE Value Function Analysis Package - Getting Started

Welcome! This package provides complete tools for analyzing the value function from your trained SafeDICE policy.

## 📋 What You Have

A complete analysis package with:
- **Main scripts**: Load weights, evaluate states, classify trajectories
- **Examples**: 5 working demonstrations 
- **Utilities**: Filter and cluster trajectories by quality
- **Documentation**: Complete API reference and guides

## 🚀 Quick Start (Choose Your Path)

### Path 1: I want to test it now (2 minutes)
```bash
# Validate that everything works
python validate_package.py

# Then run analysis with defaults
python analyze_value_function.py
```

### Path 2: I want to see examples (5 minutes)
```bash
# Run 5 complete working examples
python example_value_function_usage.py
```

### Path 3: I want to use it in my code (10 minutes)
```python
from analyze_value_function import ValueFunctionAnalyzer

# Load the value function
analyzer = ValueFunctionAnalyzer(
    'SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
)

# Evaluate states
states = ... # your states
values = analyzer.get_state_values(states)

# Classify as high/low
values, labels = analyzer.classify_states_by_value(states)

# Analyze trajectories
result = analyzer.analyze_trajectory(trajectory)
```

### Path 4: I want to filter trajectories (5 minutes)
```bash
# Analyze distribution
python filter_trajectories_by_value.py --filter-type analyze

# Create high-quality dataset
python filter_trajectories_by_value.py \
    --filter-type high_quality \
    --top-percent 25 \
    --output-dataset high_quality.pickle
```

## 📁 Files in This Package

| File | Purpose | Lines | When to Use |
|------|---------|-------|-----------|
| **analyze_value_function.py** | Main analysis module | 850+ | Always - core functionality |
| **example_value_function_usage.py** | Working examples | 400+ | First time setup, learning |
| **filter_trajectories_by_value.py** | Trajectory filtering | 400+ | Dataset curation, quality filtering |
| **validate_package.py** | Validation script | 250+ | Initial setup, troubleshooting |
| **QUICK_REFERENCE.md** | Quick start guide | 500+ | Fast lookup, common tasks |
| **VALUE_FUNCTION_ANALYSIS_README.md** | Full docs | 700+ | Complete API reference |
| **PACKAGE_SUMMARY.md** | Package overview | 300+ | Understanding architecture |
| **README_GETTING_STARTED.md** | This file | - | You are here! |

## 🎯 Common Tasks

### Task 1: Estimate values for a set of states
```python
from analyze_value_function import ValueFunctionAnalyzer
import numpy as np

analyzer = ValueFunctionAnalyzer('SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle')

states = np.random.randn(100, 60).astype(np.float32)
values = analyzer.get_state_values(states)

print(f"Min value: {values.min():.4f}")
print(f"Max value: {values.max():.4f}")
print(f"Mean value: {values.mean():.4f}")
```

### Task 2: Classify states as safe/unsafe
```python
values, safe_mask = analyzer.classify_states_by_value(
    all_states,
    method='percentile',
    threshold=0.33  # Bottom 33% = unsafe
)

safe_states = all_states[safe_mask]
unsafe_states = all_states[~safe_mask]

print(f"Safe: {len(safe_states)}, Unsafe: {len(unsafe_states)}")
```

### Task 3: Analyze a trajectory
```python
trajectory = {
    'observations': states,  # Required
    'rewards': rewards,      # Optional
    'costs': costs,          # Optional
}

result = analyzer.analyze_trajectory(trajectory)
print(result)

# Key outputs:
# - result['value_mean']: Avenue value in trajectory
# - result['value_reward_correlation']: How well value predicts reward
# - result['estimated_trajectory_quality']: Quality score
```

### Task 4: Process entire dataset
```python
analysis = analyzer.analyze_dataset(
    'SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle',
    max_trajectories=100
)

print(f"Analyzed: {analysis['num_trajectories']} trajectories")
print(f"Mean trajectory value: {analysis['value_statistics']['mean_of_means']:.4f}")

if 'value_reward_correlation_stats' in analysis:
    corr = analysis['value_reward_correlation_stats']['mean']
    print(f"Value-reward correlation: {corr:.4f}")
```

### Task 5: Select high-quality trajectories
```python
from filter_trajectories_by_value import TrajectoryFilter

filter_util = TrajectoryFilter(analyzer)

# Get top 25%
high_quality, scores = filter_util.select_high_quality_trajectories(
    trajectories,
    top_percent=25
)

print(f"Selected {len(high_quality)} high-quality trajectories")
```

### Task 6: Cluster trajectories by quality
```python
clusters = filter_util.cluster_trajectories(
    trajectories,
    num_clusters=3
)

for cluster_id, cluster_data in clusters.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Size: {cluster_data['num_trajectories']}")
    print(f"  Mean value: {cluster_data['mean_score']:.4f}")
```

## 📚 Documentation Map

```
Quick Start? 👉 QUICK_REFERENCE.md
              (500 lines, TL;DR sections)

Need Examples? 👉 example_value_function_usage.py
               (Run it directly)

Full Details? 👉 VALUE_FUNCTION_ANALYSIS_README.md
            (700 lines, complete reference)

Architecture? 👉 PACKAGE_SUMMARY.md
             (300 lines, system overview)

Need Validation? 👉 validate_package.py
                (Run to check everything)

Lost? 👉 This file, then pick appropriate path above
```

## ✅ Validation Checklist

- [ ] Have you run `validate_package.py`? (Checks everything is installed)
- [ ] Have you seen `example_value_function_usage.py`? (5 working examples)
- [ ] Have you tried the quick start above? (Pick Path 1-4)
- [ ] Have you looked at `QUICK_REFERENCE.md`? (Common solutions)
- [ ] Are you ready to integrate into your code? (Use the API)

## 🔧 API Cheat Sheet

```python
# Make analyzer
analyzer = ValueFunctionAnalyzer('weights.pickle')

# Core methods
values              = analyzer.get_state_values(states)          # Get values
values, labels      = analyzer.classify_states_by_value(states)  # Classify
result              = analyzer.analyze_trajectory(traj)          # Analyze
analysis            = analyzer.analyze_dataset('dataset.pickle') # Dataset analysis

# Visualization
analyzer.plot_value_distribution(states, save_path='fig.png')
analyzer.plot_trajectory_analysis(trajectory, save_path='fig.png')

# Filtering
from filter_trajectories_by_value import TrajectoryFilter
filter_util = TrajectoryFilter(analyzer)

filtered, scores = filter_util.select_high_quality_trajectories(trajs, 25)
clusters         = filter_util.cluster_trajectories(trajs, 3)
metadata         = filter_util.create_filtered_dataset(in_path, out_path)
```

## 🐛 Troubleshooting

**Issue**: "FileNotFoundError: weights.pickle"
```
Solution: Check path is correct, use absolute path if needed
analyze_value_function.py --weights /full/path/to/weights.pickle
```

**Issue**: "No critic parameters found"  
```
Solution: Ensure weights file is from SafeDICE training, not corrupted
```

**Issue**: Poor value-reward correlation
```
Solution: Try different checkpoint, check training convergence, 
see VALUE_FUNCTION_ANALYSIS_README.md for troubleshooting
```

**Issue**: Slow processing
```
Solution: Reduce max_trajectories, enable GPU, batch processing
See QUICK_REFERENCE.md "Performance Tips"
```

## 📊 What You Can Do

With this package you can:

✅ **Evaluate** - Get value estimates for any state  
✅ **Classify** - Categorize states as high/low value  
✅ **Analyze** - Understand trajectory quality  
✅ **Understand** - See value-reward correlations  
✅ **Filter** - Select high-quality trajectories  
✅ **Cluster** - Group trajectories by quality  
✅ **Visualize** - Plot distributions and trends  
✅ **Export** - Save filtered datasets  
✅ **Integrate** - Use in your own code  

## 🎓 Learning Path

1. **Beginner** (20 minutes)
   - Run `validate_package.py` 
   - Run `example_value_function_usage.py`
   - Read `QUICK_REFERENCE.md`

2. **Intermediate** (1 hour)
   - Use API for your states
   - Try basic filtering
   - Read relevant sections of docs

3. **Advanced** (2+ hours)
   - Read `VALUE_FUNCTION_ANALYSIS_README.md` completely
   - Implement custom analysis
   - Integrate into your pipeline
   - Create custom visualizations

## 📞 Getting Help

### For Quick Questions
👉 Check `QUICK_REFERENCE.md` first (most questions answered there)

### For API Questions  
👉 See `VALUE_FUNCTION_ANALYSIS_README.md` - complete reference with examples

### For Examples
👉 Run `example_value_function_usage.py` or read it

### For Errors
👉 See "Troubleshooting" section of `VALUE_FUNCTION_ANALYSIS_README.md`

### For Architecture 
👉 Read `PACKAGE_SUMMARY.md`

## 💾 Expected File Sizes

- Script files: ~50-100 KB each
- Documentation: ~50-100 KB each  
- Weights file: ~200-500 MB (required for analysis)
- Dataset file: ~500 MB - 2 GB (required for dataset analysis)

## 🚀 Next Steps

Pick one option below:

**Option A: I want to try it immediately**
```bash
python validate_package.py  # Check everything works
python analyze_value_function.py  # Run basic analysis
```

**Option B: I want to see what's possible**
```bash
python example_value_function_usage.py  # Run 5 examples
```

**Option C: I want to use it in my code**
```python
from analyze_value_function import ValueFunctionAnalyzer
analyzer = ValueFunctionAnalyzer('weights.pickle')
values = analyzer.get_state_values(states)
```

**Option D: I want to understand first**
Read `QUICK_REFERENCE.md` (10 minutes)
Then pick Option A, B, or C above

---

## 📝 Notes

- All scripts are **production-ready** with error handling
- All scripts are **well-documented** with docstrings
- All scripts work **standalone** - no additional setup needed
- All documentation is **comprehensive** - from quick start to advanced
- All code is **tested** - examples are verified working

## 🎯 You're Ready!

You have everything needed to:
1. Load SafeDICE value functions
2. Evaluate states
3. Classify trajectories by quality
4. Create high-quality datasets
5. Understand policy alignment
6. Integrate with your own code

**Start with one of the paths above, then explore! 🚀**

Questions? Check the appropriate documentation file listed in the "Documentation Map" section above.

---

**Happy analyzing!** 🎯
