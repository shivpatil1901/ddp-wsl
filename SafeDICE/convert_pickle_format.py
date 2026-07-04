#!/usr/bin/env python
"""
Convert pickle files to match expected format with keys:
['init_states', 'states', 'actions', 'next_states', 'costs', 'rewards', 'dones']
"""
import pickle
import numpy as np
import os

def convert_pickle_format(input_path, output_path, verbose=True):
    """
    Load pickle and convert to standard format.
    Maps common key names to expected format.
    """
    
    if verbose:
        print(f"\nLoading: {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    
    original_keys = set(data.keys())
    if verbose:
        print(f"Original keys: {list(original_keys)}")
    
    # Expected keys
    expected_keys = {'init_states', 'states', 'actions', 'next_states', 'costs', 'rewards', 'dones'}
    
    # If already has all expected keys, just copy
    if expected_keys.issubset(original_keys):
        if verbose:
            print("✓ Already in correct format")
        converted = data
    else:
        # Map common variations
        converted = {}
        
        # Map states
        if 'observations' in data:
            converted['states'] = data['observations']
        elif 'states' in data:
            converted['states'] = data['states']
        elif 'obs' in data:
            converted['states'] = data['obs']
        else:
            raise KeyError("Cannot find 'states' - tried: observations, states, obs")
        
        # Map init_states (usually first state of each trajectory)
        if 'init_states' in data:
            converted['init_states'] = data['init_states']
        elif 'initial_states' in data:
            converted['init_states'] = data['initial_states']
        else:
            # Create init_states from first state if needed
            if verbose:
                print("⚠ Creating init_states from first state of trajectories")
            converted['init_states'] = converted['states'][::len(converted['states'])//1000] if 'states' in converted else None
        
        # Map next_states
        if 'next_states' in data:
            converted['next_states'] = data['next_states']
        elif 'next_observations' in data:
            converted['next_states'] = data['next_observations']
        elif 'next_obs' in data:
            converted['next_states'] = data['next_obs']
        else:
            raise KeyError("Cannot find 'next_states'")
        
        # Map actions
        if 'actions' in data:
            converted['actions'] = data['actions']
        else:
            raise KeyError("Missing 'actions'")
        
        # Map rewards
        if 'rewards' in data:
            converted['rewards'] = data['rewards']
        else:
            raise KeyError("Missing 'rewards'")
        
        # Map costs
        if 'costs' in data:
            converted['costs'] = data['costs']
        elif 'cost' in data:
            converted['costs'] = data['cost']
        else:
            if verbose:
                print("⚠ Creating zero costs (not found in original)")
            converted['costs'] = np.zeros_like(data['rewards'])
        
        # Map dones
        if 'dones' in data:
            converted['dones'] = data['dones']
        elif 'done' in data:
            converted['dones'] = data['done']
        else:
            raise KeyError("Missing 'dones'")
    
    # Verify all keys are present
    missing = expected_keys - set(converted.keys())
    if missing:
        raise KeyError(f"Missing keys: {missing}")
    
    if verbose:
        print(f"Converted keys: {list(converted.keys())}")
        for key in expected_keys:
            if key in converted and hasattr(converted[key], 'shape'):
                print(f"  {key}: shape={converted[key].shape}, dtype={converted[key].dtype}")
    
    # Save converted data
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(converted, f)
    
    if verbose:
        print(f"✓ Saved to: {output_path}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert pickle files to standard format')
    parser.add_argument('input', help='Input pickle file path')
    parser.add_argument('output', nargs='?', help='Output pickle file path (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup of original')
    
    args = parser.parse_args()
    
    output = args.output if args.output else args.input
    
    # Create backup if modifying in place
    if output == args.input and not args.no_backup:
        backup = args.input + '.bak'
        import shutil
        shutil.copy(args.input, backup)
        print(f"Created backup: {backup}")
    
    try:
        convert_pickle_format(args.input, output)
        print("✓ Conversion successful!")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)
