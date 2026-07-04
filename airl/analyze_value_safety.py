import gym
import safety_gym
import numpy as np
import matplotlib
# --- FIX: Force Headless Mode (Must be before pyplot import) ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pickle
import os

# ==========================================
# 1. Configuration
# ==========================================
ROBOT_NAME = 'Point'
TASK_NAME = 'Goal1'
ENV_NAME = f'Safexp-{ROBOT_NAME}{TASK_NAME}-v0'
PICKLE_PATH = "../SafeDICE/weights/antidice_PointGoal1_seed0_20260206_184622_iter1000000.pickle"

# ==========================================
# 2. Load Value Function
# ==========================================
def load_value_function(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    params = data['training_state']['critic_params']
    
    w1, b1 = params[0][1], params[1][1]
    w2, b2 = params[2][1], params[3][1]
    w3 = params[4][1]
    b3 = params[5][1] if len(params) > 5 else np.zeros(1)

    def predict_value(state):
        s = np.array(state).flatten()
        h1 = np.maximum(np.dot(s, w1) + b1, 0)
        h2 = np.maximum(np.dot(h1, w2) + b2, 0)
        return float(np.dot(h2, w3) + b3)

    return predict_value

# ==========================================
# 3. Analyze Environment States
# ==========================================
def find_unsafe_states(env_name, value_fn, num_steps=10000, percentile_threshold=15):
    print(f"\nInitializing Environment: {env_name}")
    env = gym.make(env_name)
    
    states, values, true_costs = [], [], []
    
    print(f"Collecting {num_steps} steps...")
    obs = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        v = value_fn(obs)
        
        states.append(obs)
        values.append(v)
        
        obs, _, done, info = env.step(action)
        true_costs.append(info.get('cost', 0))
        
        if done: obs = env.reset()
            
    states = np.array(states)
    values = np.array(values)
    true_costs = np.array(true_costs)
    
    # Analysis
    cutoff = np.percentile(values, percentile_threshold)
    unsafe_mask = values < cutoff
    unsafe_states = states[unsafe_mask]
    
    print("\n" + "="*40)
    print(f"SAFETY VALUE ANALYSIS ({percentile_threshold}th Percentile)")
    print("="*40)
    print(f"Value Cutoff:       {cutoff:.4f}")
    print(f"States Scanned:     {len(states)}")
    print(f"Flagged 'Bad':      {len(unsafe_states)}")
    
    avg_bad = np.mean(true_costs[unsafe_mask])
    avg_good = np.mean(true_costs[~unsafe_mask])
    
    print(f"\nVALIDATION (True Env Cost):")
    print(f"Avg Cost in 'Bad':  {avg_bad:.4f}")
    print(f"Avg Cost in 'Good': {avg_good:.4f}")

    # --- Plotting (Save to File) ---
    plt.figure(figsize=(10, 6))
    plt.hist(values[~unsafe_mask], bins=50, color='green', alpha=0.6, label='High Value (Safe)')
    plt.hist(values[unsafe_mask], bins=50, color='red', alpha=0.6, label='Low Value (Unsafe)')
    plt.axvline(cutoff, color='black', linestyle='--', label=f'Threshold ({cutoff:.2f})')
    
    plt.xlabel("Learned Value V(s)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of State Values in {env_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save instead of show
    filename = "safety_value_analysis.png"
    plt.savefig(filename)
    print(f"\nPlot saved to: {os.path.abspath(filename)}")
    
    return unsafe_states

if __name__ == "__main__":
    value_fn = load_value_function(PICKLE_PATH)
    find_unsafe_states(ENV_NAME, value_fn)