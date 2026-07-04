#!/usr/bin/env python
"""
Convert ppo_PointGoal1_s0.pickle from trajectory list format to flattened array format
to match ppo_lagrangian_PointGoal1_s0.pickle format.

Source format: {'trajectories': [traj1, traj2, ...], 'metadata': {...}}
Target format: {'init_states', 'states', 'actions', 'next_states', 'costs', 'rewards', 'dones'}
"""
import pickle
import numpy as np
import os

def convert_trajectory_format(input_path, output_path, verbose=True):
    """
    Convert trajectory list format to flattened array format.
    """
    
    if verbose:
        print(f"\nLoading: {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, dict) or 'trajectories' not in data:
        raise TypeError(f"Expected dict with 'trajectories' key, got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    
    trajectories = data['trajectories']
    if verbose:
        print(f"Found {len(trajectories)} trajectories")
        print(f"Metadata keys: {list(data.get('metadata', {}).keys())}")
    
    # Infer trajectory format from first trajectory
    first_traj = trajectories[0]
    if verbose:
        if isinstance(first_traj, dict):
            print(f"Trajectory format: dict with keys {list(first_traj.keys())}")
        else:
            print(f"Trajectory format: {type(first_traj)}")
    
    # Extract and flatten data
    init_states_list = []
    states_list = []
    actions_list = []
    next_states_list = []
    costs_list = []
    rewards_list = []
    dones_list = []
    
    for i, traj in enumerate(trajectories):
        if isinstance(traj, dict):
            # Dict format: {'states': ..., 'actions': ..., etc.}
            states = traj.get('observations') if traj.get('observations') is not None else traj.get('states')
            actions = traj.get('actions')
            next_states = traj.get('next_observations') if traj.get('next_observations') is not None else traj.get('next_states')
            costs = traj.get('costs') if traj.get('costs') is not None else np.zeros((len(states), 1))
            rewards = traj.get('rewards')
            dones = traj.get('dones')
            
            if states is None:
                raise KeyError(f"Trajectory {i}: Cannot find states (tried: observations, states)")
            if actions is None:
                raise KeyError(f"Trajectory {i}: Missing actions")
            if next_states is None:
                # Create next_states by shifting states by 1 and using last state for last timestep
                if verbose and i == 0:
                    print("⚠ Creating next_states by shifting states")
                next_states = np.vstack([states[1:], states[-1:]])
            if rewards is None:
                raise KeyError(f"Trajectory {i}: Missing rewards")
            if dones is None:
                raise KeyError(f"Trajectory {i}: Missing dones")
            
        else:
            raise TypeError(f"Unsupported trajectory format: {type(traj)}")
        
        # Ensure 1D costs/rewards/dones become 2D
        if len(costs.shape) == 1:
            costs = costs[:, np.newaxis]
        if len(rewards.shape) == 1:
            rewards = rewards[:, np.newaxis]
        if len(dones.shape) == 1:
            dones = dones[:, np.newaxis]
        
        # Store init state (first state)
        init_states_list.append(states[0:1])
        
        # Store trajectory data
        states_list.append(states)
        actions_list.append(actions)
        next_states_list.append(next_states)
        costs_list.append(costs)
        rewards_list.append(rewards)
        dones_list.append(dones)
        
        if (i + 1) % max(1, len(trajectories) // 10) == 0:
            print(f"  Processed {i+1}/{len(trajectories)} trajectories")
    
    # Concatenate all trajectories
    init_states = np.concatenate(init_states_list, axis=0)
    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    next_states = np.concatenate(next_states_list, axis=0)
    costs = np.concatenate(costs_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    dones = np.concatenate(dones_list, axis=0)
    
    if verbose:
        print(f"\nFlattened data:")
        print(f"  init_states: shape={init_states.shape}, dtype={init_states.dtype}")
        print(f"  states: shape={states.shape}, dtype={states.dtype}")
        print(f"  actions: shape={actions.shape}, dtype={actions.dtype}")
        print(f"  next_states: shape={next_states.shape}, dtype={next_states.dtype}")
        print(f"  costs: shape={costs.shape}, dtype={costs.dtype}")
        print(f"  rewards: shape={rewards.shape}, dtype={rewards.dtype}")
        print(f"  dones: shape={dones.shape}, dtype={dones.dtype}")
    
    # Create output dict
    converted = {
        'init_states': init_states,
        'states': states,
        'actions': actions,
        'next_states': next_states,
        'costs': costs,
        'rewards': rewards,
        'dones': dones
    }
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(converted, f)
    
    if verbose:
        print(f"\n✓ Saved to: {output_path}")
    
    return converted

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert trajectory list format to flattened array format')
    parser.add_argument('input', help='Input pickle file (trajectory format)')
    parser.add_argument('--output', help='Output pickle file (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup of original')
    
    args = parser.parse_args()
    
    output = args.output if args.output else args.input
    
    # Create backup if modifying in place
    if output == args.input and not args.no_backup:
        backup = args.input + '.bak'
        import shutil
        if os.path.exists(args.input):
            shutil.copy(args.input, backup)
            print(f"Created backup: {backup}")
    
    try:
        convert_trajectory_format(args.input, output)
        print("\n✓ Conversion successful!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
