import gym
import safety_gym  # Registers Safexp-PointGoal1-v0
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode for WSL
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import os

# ==========================================
# 1. The Wrapper Class (Same as before)
# ==========================================
class SafeDICERewardWrapper(gym.Wrapper):
    def __init__(self, env, pickle_path, scale=1.0, shift=0.0):
        super().__init__(env)
        self.scale = scale
        self.shift = shift
        self.last_state = None
        self._load_cost_network(pickle_path)

    def _load_cost_network(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        params = data['training_state']['cost_params']
        self.w1, self.b1 = params[0][1], params[1][1]
        self.w2, self.b2 = params[2][1], params[3][1]
        self.w3 = params[4][1]
        self.b3 = params[5][1] if len(params) > 5 else np.zeros(1)

    def _predict_logit(self, state, action):
        s = np.array(state).flatten(); a = np.array(action).flatten()
        x = np.concatenate([s, a])
        h1 = np.maximum(np.dot(x, self.w1) + self.b1, 0)
        h2 = np.maximum(np.dot(h1, self.w2) + self.b2, 0)
        return float(np.dot(h2, self.w3) + self.b3)

    def reset(self, **kwargs):
        self.last_state = self.env.reset(**kwargs)
        return self.last_state

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)
        
        # Calculate Learned Reward
        logit = self._predict_logit(self.last_state, action)
        
        # Transform: Stable Log-Sigmoid
        adjusted_logit = logit + self.shift
        learned_reward = -np.logaddexp(0, adjusted_logit) * self.scale
        
        # Log for comparison
        info['original_reward'] = env_reward
        info['safedice_reward'] = learned_reward
        info['safedice_logit'] = logit
        
        self.last_state = next_state
        return next_state, learned_reward, done, info

# ==========================================
# 2. Comparison Logic
# ==========================================
def compare_rewards(env_name, pickle_path, num_steps=5000):
    print(f"\nInitializing Comparison for {env_name}...")
    
    # Create Wrapped Env
    base_env = gym.make(env_name)
    env = SafeDICERewardWrapper(base_env, pickle_path, scale=10.0)
    
    env_rewards = []
    safedice_rewards = []
    safety_costs = []
    
    obs = env.reset()
    print(f"Collecting {num_steps} steps...")

    for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"Step {i+1}:")
            print(f"  > Original Env Reward: {info['original_reward']:.4f}")
            print(f"  > SafeDICE Logit:      {info['safedice_logit']:.4f} (High=Unsafe)")
            print(f"  > Final Used Reward:   {reward:.4f}")
            
            if done:
                obs = env.reset()
    
    for _ in range(num_steps):
        action = env.action_space.sample()
        
        # Step returns the SafeDICE reward
        obs, sd_reward, done, info = env.step(action)
        
        # Info contains the Original reward
        env_rewards.append(info['original_reward'])
        safedice_rewards.append(sd_reward)
        safety_costs.append(info.get('cost', 0))
        
        if done:
            obs = env.reset()
            
    # Convert to arrays
    env_rewards = np.array(env_rewards)
    safedice_rewards = np.array(safedice_rewards)
    safety_costs = np.array(safety_costs)
    
    # --- Statistics ---
    print("\n" + "="*40)
    print("REWARD COMPARISON STATISTICS")
    print("="*40)
    
    # 1. Correlation with Task Reward
    corr_task, _ = stats.pearsonr(env_rewards, safedice_rewards)
    print(f"Correlation (SafeDICE vs Env Reward): {corr_task:.4f}")
    
    # 2. Correlation with Safety Cost
    # High Cost should equal Low SafeDICE Reward (Negative Correlation)
    if np.sum(safety_costs) > 0:
        corr_safe, _ = stats.pointbiserialr(safety_costs > 0, safedice_rewards)
        print(f"Correlation (SafeDICE vs Safety Cost): {corr_safe:.4f}")
    else:
        print("Warning: No safety costs incurred during collection.")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Scatter (Env vs SafeDICE)
    plt.subplot(1, 2, 1)
    plt.scatter(env_rewards, safedice_rewards, alpha=0.3, s=5, c='blue')
    plt.xlabel("Original Env Reward (Task)")
    plt.ylabel("SafeDICE Learned Reward")
    plt.title(f"Reward Alignment (r={corr_task:.2f})")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Histogram by Safety
    plt.subplot(1, 2, 2)
    safe_mask = safety_costs == 0
    unsafe_mask = safety_costs > 0
    
    plt.hist(safedice_rewards[safe_mask], bins=40, color='green', alpha=0.6, label='Safe Steps', density=True)
    if np.sum(unsafe_mask) > 0:
        plt.hist(safedice_rewards[unsafe_mask], bins=40, color='red', alpha=0.6, label='Unsafe Steps', density=True)
    plt.xlabel("SafeDICE Learned Reward")
    plt.title("Separation of Safe vs Unsafe")
    plt.legend()
    
    filename = "reward_comparison.png"
    plt.savefig(filename)
    print(f"\nComparison plot saved to: {os.path.abspath(filename)}")

if __name__ == "__main__":
    PICKLE_PATH = "../SafeDICE/weights/antidice_PointGoal1_seed0_20260206_184622_iter1000000.pickle"
    ENV_NAME = "Safexp-PointGoal1-v0"
    
    compare_rewards(ENV_NAME, PICKLE_PATH)