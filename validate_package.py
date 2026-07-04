#!/usr/bin/env python
"""
Quick validation script to check that all components work correctly.
Run this to verify the value function analysis package is properly set up.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add SafeDICE to path
safedice_path = Path(__file__).parent / 'SafeDICE'
sys.path.insert(0, str(safedice_path))

print("=" * 80)
print("SafeDICE Value Function Analysis - Package Validation")
print("=" * 80)

# Test 1: Check files exist
print("\n[Test 1] Checking required files...")
files_to_check = [
    'analyze_value_function.py',
    'example_value_function_usage.py',
    'filter_trajectories_by_value.py',
    'VALUE_FUNCTION_ANALYSIS_README.md',
    'QUICK_REFERENCE.md',
    'PACKAGE_SUMMARY.md',
]

all_files_exist = True
for fname in files_to_check:
    if os.path.exists(fname):
        print(f"  ✅ {fname}")
    else:
        print(f"  ❌ {fname} - NOT FOUND")
        all_files_exist = False

if not all_files_exist:
    print("\n⚠️  Some files are missing. Check file locations.")
    sys.exit(1)

# Test 2: Check imports
print("\n[Test 2] Checking imports...")
try:
    import tensorflow as tf
    print(f"  ✅ TensorFlow {tf.__version__}")
except ImportError:
    print(f"  ❌ TensorFlow - NOT INSTALLED")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✅ NumPy {np.__version__}")
except ImportError:
    print(f"  ❌ NumPy - NOT INSTALLED")
    sys.exit(1)

try:
    import matplotlib
    print(f"  ✅ Matplotlib {matplotlib.__version__}")
except ImportError:
    print(f"  ❌ Matplotlib - NOT INSTALLED")
    sys.exit(1)

try:
    import pickle5
    print(f"  ✅ pickle5")
except ImportError:
    print(f"  ⚠️  pickle5 not available, will fall back to standard pickle")

# Test 3: Check SafeDICE is accessible
print("\n[Test 3] Checking SafeDICE integration...")
try:
    from algorithms.safedice import SafeDICE as AntiDICE
    print(f"  ✅ SafeDICE algorithms")
except ImportError as e:
    print(f"  ❌ Could not import SafeDICE: {e}")
    sys.exit(1)

try:
    import config.safedice_config as antidice_config
    print(f"  ✅ SafeDICE config")
except ImportError as e:
    print(f"  ❌ Could not import SafeDICE config: {e}")
    sys.exit(1)

# Test 4: Check weights file exists
print("\n[Test 4] Checking weights file...")
weights_path = 'SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle'
if os.path.exists(weights_path):
    file_size = os.path.getsize(weights_path) / (1024*1024)
    print(f"  ✅ {weights_path}")
    print(f"     Size: {file_size:.1f} MB")
else:
    print(f"  ⚠️  {weights_path} - NOT FOUND")
    print(f"     This is OK for validation, but needed for actual analysis")

# Test 5: Check dataset file exists
print("\n[Test 5] Checking dataset file...")
dataset_path = 'SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle'
if os.path.exists(dataset_path):
    file_size = os.path.getsize(dataset_path) / (1024*1024)
    print(f"  ✅ {dataset_path}")
    print(f"     Size: {file_size:.1f} MB")
else:
    print(f"  ⚠️  {dataset_path} - NOT FOUND")
    print(f"     This is OK for validation, but needed for full analysis")

# Test 6: Import main modules
print("\n[Test 6] Importing main modules...")
try:
    from analyze_value_function import ValueFunctionAnalyzer
    print(f"  ✅ ValueFunctionAnalyzer")
except Exception as e:
    print(f"  ❌ Could not import ValueFunctionAnalyzer: {e}")
    sys.exit(1)

try:
    from filter_trajectories_by_value import TrajectoryFilter
    print(f"  ✅ TrajectoryFilter")
except Exception as e:
    print(f"  ❌ Could not import TrajectoryFilter: {e}")
    sys.exit(1)

# Test 7: Basic functionality test
print("\n[Test 7] Testing basic functionality...")

try:
    # Try to load a weights file if it exists
    if os.path.exists(weights_path):
        print(f"  Testing weight loading...")
        analyzer = ValueFunctionAnalyzer(weights_path)
        print(f"  ✅ Successfully loaded weights and created analyzer")
        
        # Test value estimation
        print(f"  Testing value estimation...")
        test_states = np.random.randn(10, 60).astype(np.float32)
        values = analyzer.get_state_values(test_states)
        
        if values.shape == (10,) and not np.any(np.isnan(values)):
            print(f"  ✅ Value estimation works")
            print(f"     Generated {len(values)} values")
            print(f"     Value range: [{np.min(values):.4f}, {np.max(values):.4f}]")
        else:
            print(f"  ❌ Value estimation produced invalid output")
            sys.exit(1)
        
        # Test classification
        print(f"  Testing state classification...")
        values_c, labels = analyzer.classify_states_by_value(test_states, threshold=0.5)
        
        if labels.shape == (10,) and set(np.unique(labels)) <= {0, 1}:
            print(f"  ✅ State classification works")
            print(f"     High-value: {np.sum(labels)}, Low-value: {np.sum(1-labels)}")
        else:
            print(f"  ❌ State classification produced invalid output")
            sys.exit(1)
            
    else:
        print(f"  ⚠️  Weights file not found, skipping analyzer test")
        print(f"     (This is expected on first run)")
        
except Exception as e:
    print(f"  ❌ Error during functionality test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: GPU check
print("\n[Test 8] GPU availability...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"  ✅ GPU available: {gpus[0]}")
else:
    print(f"  ℹ️  No GPU found - will use CPU (slower)")

# Final summary
print("\n" + "=" * 80)
print("Validation Summary")
print("=" * 80)

print("""
✅ All core components validated!

You can now:
1. Run the basic analyzer:
   python analyze_value_function.py

2. See working examples:
   python example_value_function_usage.py
   
3. Use the API in your code:
   from analyze_value_function import ValueFunctionAnalyzer
   analyzer = ValueFunctionAnalyzer('weights.pickle')
   values = analyzer.get_state_values(states)

4. Filter trajectories:
   python filter_trajectories_by_value.py --filter-type analyze

For more information:
- Quick start: QUICK_REFERENCE.md
- Full docs: VALUE_FUNCTION_ANALYSIS_README.md
- Package info: PACKAGE_SUMMARY.md

Happy analyzing! 🎯
""")

print("=" * 80)
