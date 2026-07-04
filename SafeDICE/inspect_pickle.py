#!/usr/bin/env python
import pickle5 as pickle
import os

# Check what's in both pickle files
files = [
    './dataset/safetygym/ppo_PointGoal1_s0.pickle',
    './dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle'
]

for fpath in files:
    if os.path.exists(fpath):
        print(f"\n{'='*60}")
        print(f"File: {fpath}")
        print('='*60)
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
                for key in data.keys():
                    if hasattr(data[key], 'shape'):
                        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                    else:
                        print(f"  {key}: type={type(data[key])}, len={len(data[key]) if hasattr(data[key], '__len__') else 'N/A'}")
            else:
                print(f"Type: {type(data)}")
        except Exception as e:
            print(f"Error loading: {e}")
    else:
        print(f"File not found: {fpath}")
