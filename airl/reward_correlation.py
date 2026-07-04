# import numpy as np
# import tensorflow as tf
# import pickle5 as pickle
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, spearmanr

# # Import your discriminator class from your training script
# # Ensure airl_safedice.py is in the same folder or in path
# from airl_safedice import AIRLDiscriminator

# def check_offline_correlation(expert_path, checkpoint_path, obs_dim, act_dim):
#     # 1. Load Expert Data
#     print(f"Loading expert data from: {expert_path}")
#     with open(expert_path, 'rb') as f:
#         expert_demo = pickle.load(f)
    
#     e_states = expert_demo['states']
#     e_actions = expert_demo['actions']
#     e_rewards = expert_demo['rewards'] # Ground truth from env

#     # 2. Initialize and Load Learned Reward Model
#     print("Loading learned AIRL reward model...")
#     discriminator = AIRLDiscriminator(obs_dim, act_dim)
#     # We need to call the model once to build the variables before loading weights
#     dummy_s = tf.zeros((1, obs_dim))
#     dummy_a = tf.zeros((1, act_dim))
#     discriminator.g_network(tf.concat([dummy_s, dummy_a], axis=-1))
    
#     discriminator.load_weights(checkpoint_path)

#     # 3. Compute Learned Rewards g(s, a)
#     print("Computing learned rewards for expert transitions...")
#     # Process in batches to avoid GPU OOM if dataset is huge
#     batch_size = 5000
#     learned_rewards = []
    
#     for i in range(0, len(e_states), batch_size):
#         s_batch = tf.constant(e_states[i:i+batch_size], dtype=tf.float32)
#         a_batch = tf.constant(e_actions[i:i+batch_size], dtype=tf.float32)
#         sa_batch = tf.concat([s_batch, a_batch], axis=-1)
        
#         g_sa = discriminator.g_network(sa_batch).numpy().flatten()
#         learned_rewards.extend(g_sa)
    
#     learned_rewards = np.array(learned_rewards)
#     e_rewards = np.array(e_rewards).flatten()

#     # 4. Calculate Correlation Metrics
#     # Pearson: Linear relationship
#     p_corr, _ = pearsonr(e_rewards, learned_rewards)
#     # Spearman: Rank/Order relationship (Often more important for RL)
#     s_corr, _ = spearmanr(e_rewards, learned_rewards)

#     print("\n" + "="*30)
#     print(f"RESULTS FOR {len(e_rewards)} TRANSITIONS")
#     print(f"Pearson Correlation:  {p_corr:.4f}")
#     print(f"Spearman Correlation: {s_corr:.4f}")
#     print("="*30)

#     # 5. Plotting
#     plt.figure(figsize=(10, 6))
#     plt.scatter(e_rewards[::10], learned_rewards[::10], alpha=0.3, s=2) # Sample every 10th for clarity
#     plt.title(f"Env Reward vs. Learned AIRL Reward (Spearman: {s_corr:.3f})")
#     plt.xlabel("Environment Ground Truth Reward")
#     plt.ylabel("Learned Reward g(s, a)")
#     plt.grid(True)
#     plt.savefig("reward_correlation_plot.png")
#     print("Plot saved as reward_correlation_plot.png")
#     plt.show()

# if __name__ == "__main__":
#     # Update these paths to match your local setup
#     EXPERT_PATH = "../SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
#     CHECKPOINT_PATH = "./airl_results/reward_model_final.ckpt"
    
#     # Dimensions for PointGoal1
#     OBS_DIM = 60 
#     ACT_DIM = 2

#     check_offline_correlation(EXPERT_PATH, CHECKPOINT_PATH, OBS_DIM, ACT_DIM)



# import numpy as np
# import tensorflow as tf
# import pickle5 as pickle
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, spearmanr
# import os

# from airl_safedice import AIRLDiscriminator

# def canonicalize_reward(rewards, states, next_states, gamma=0.99):
#     """
#     Approximates the canonical form of the reward to remove shaping.
#     C(s,a,s') = R(s,a,s') + gamma*E[V(s')] - V(s)
#     """
#     # In a simplified offline setting, we use the mean reward as a baseline
#     return rewards + gamma * np.mean(rewards) - np.mean(rewards)

# def check_offline_metrics(expert_path, checkpoint_path, obs_dim, act_dim, gamma=0.99):
#     # 1. Load Data
#     print(f"📂 Loading expert data...")
#     with open(expert_path, 'rb') as f:
#         try:
#             import pickle5 as pkl
#             expert_demo = pkl.load(f)
#         except ImportError:
#             expert_demo = pickle.load(f)
    
#     e_states = expert_demo['states']
#     e_actions = expert_demo['actions']
#     e_next_states = expert_demo['next_states']
#     e_rewards = np.array(expert_demo['rewards']).flatten()
#     e_costs = np.array(expert_demo['costs']).flatten() if 'costs' in expert_demo else None

#     # 2. Load Model
#     discriminator = AIRLDiscriminator(obs_dim, act_dim, gamma=gamma)
#     dummy_s, dummy_a, dummy_ns = tf.zeros((1, obs_dim)), tf.zeros((1, act_dim)), tf.zeros((1, obs_dim))
#     _ = discriminator(dummy_s, dummy_a, dummy_ns)
#     discriminator.load_weights(checkpoint_path)

#     # 3. Compute Batch Rewards
#     print("🧮 Computing rewards...")
#     batch_size = 5000
#     learned_g = []
#     learned_f = []
    
#     for i in range(0, len(e_states), batch_size):
#         s_batch = tf.constant(e_states[i:i+batch_size], dtype=tf.float32)
#         a_batch = tf.constant(e_actions[i:i+batch_size], dtype=tf.float32)
#         ns_batch = tf.constant(e_next_states[i:i+batch_size], dtype=tf.float32)
        
#         sa_batch = tf.concat([s_batch, a_batch], axis=-1)
#         g_val = discriminator.g_network(sa_batch).numpy().flatten()
#         learned_g.extend(g_val)
        
#         f_val = discriminator(s_batch, a_batch, ns_batch).numpy().flatten()
#         learned_f.extend(f_val)
    
#     learned_g = np.array(learned_g)
#     learned_f = np.array(learned_f)

#     # 4. Standard & EPIC Metrics
#     can_env = canonicalize_reward(e_rewards, e_states, e_next_states, gamma)
#     can_learned = canonicalize_reward(learned_g, e_states, e_next_states, gamma)
#     epic_corr, _ = pearsonr(can_env, can_learned)
    
#     g_spearman, _ = spearmanr(e_rewards, learned_g)
#     f_spearman, _ = spearmanr(e_rewards, learned_f)

#     print("\n" + "="*45)
#     print(f"METRICS FOR {len(e_rewards)} TRANSITIONS")
#     print("-" * 45)
#     print(f"Spearman g(s, a):    {g_spearman:.4f}")
#     print(f"Spearman f(s,a,s'):  {f_spearman:.4f}")
#     print(f"EPIC Correlation:    {epic_corr:.4f}")
#     print(f"EPIC Distance:       {1 - epic_corr:.4f}")
    
#     # 5. Safety Correlation (Discriminator should penalize costs)
#     if e_costs is not None:
#         cost_corr, _ = spearmanr(e_costs, learned_g)
#         print(f"Correlation w/ Cost: {cost_corr:.4f} (Should be negative)")
#     print("="*45)

#     # 6. Top-K Transition Analysis
#     sorted_idx = np.argsort(learned_g)[::-1]
#     print("\n🔝 TOP 5 EXPERT TRANSITIONS (By Learned Reward):")
#     for i in range(5):
#         idx = sorted_idx[i]
#         print(f"Rank {i+1} | Learned: {learned_g[idx]:.3f} | Env: {e_rewards[idx]:.3f} | Cost: {e_costs[idx] if e_costs is not None else 'N/A'}")

#     # 7. Bottom-K Transition Analysis (The "Least Expert" behavior)
#     print("\n🔻 BOTTOM 5 EXPERT TRANSITIONS (By Learned Reward):")
#     for i in range(1, 6):
#         idx = sorted_idx[-i]
#         print(f"Rank {len(learned_g)-i+1} | Learned: {learned_g[idx]:.3f} | Env: {e_rewards[idx]:.3f} | Cost: {e_costs[idx] if e_costs is not None else 'N/A'}")

# if __name__ == "__main__":
#     EXPERT_PATH = "../SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
#     CHECKPOINT_PATH = "./airl_results/reward_model_final.ckpt"
#     check_offline_metrics(EXPERT_PATH, CHECKPOINT_PATH, obs_dim=60, act_dim=2)



import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle5 as pickle
from airl_safedice import AIRLDiscriminator

def acid_test_goal_distance(expert_path, checkpoint_path, obs_dim=60, act_dim=2):
    # 1. Load Data
    with open(expert_path, 'rb') as f:
        expert_demo = pickle.load(f)
    
    states = expert_demo['states']
    actions = expert_demo['actions']

    # 2. Load Model
    discriminator = AIRLDiscriminator(obs_dim, act_dim)
    # Build variables
    _ = discriminator.g_network(tf.zeros((1, obs_dim + act_dim)))
    discriminator.load_weights(checkpoint_path)

    # 3. Identify Goal Distance Feature
    # In SafetyGym, indices 0 and 1 are typically the Goal XY (relative to robot)
    # We calculate Euclidean distance: sqrt(x^2 + y^2)
    goal_dist = np.linalg.norm(states[:, :2], axis=1)

    # 4. Compute Learned Rewards (Batch Processing)
    print("Computing learned rewards...")
    batch_size = 5000
    learned_rewards = []
    for i in range(0, len(states), batch_size):
        s_batch = tf.constant(states[i:i+batch_size], dtype=tf.float32)
        a_batch = tf.constant(actions[i:i+batch_size], dtype=tf.float32)
        sa_batch = tf.concat([s_batch, a_batch], axis=-1)
        
        g_sa = discriminator.g_network(sa_batch).numpy().flatten()
        learned_rewards.extend(g_sa)
    
    learned_rewards = np.array(learned_rewards)

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    # Sample subset for cleaner plot
    idx = np.random.choice(len(goal_dist), 5000, replace=False)
    
    plt.scatter(goal_dist[idx], learned_rewards[idx], alpha=0.4, color='purple', s=10)
    
    # Add a trend line
    z = np.polyfit(goal_dist[idx], learned_rewards[idx], 1)
    p = np.poly1d(z)
    plt.plot(goal_dist[idx], p(goal_dist[idx]), "r--", label="Trend Line")

    plt.title("Acid Test: Learned Reward vs. Goal Distance")
    plt.xlabel("Relative Distance to Goal (Lower = Closer)")
    plt.ylabel("Learned Reward g(s, a)")
    plt.gca().invert_xaxis()  # Invert so moving "forward" (closer) is left to right
    plt.legend()
    plt.grid(True)
    plt.savefig("acid_test_distance.png")
    print("Plot saved as acid_test_distance.png")
    # plt.show()

if __name__ == "__main__":
    EXPERT_PATH = "../SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
    CHECKPOINT_PATH = "./airl_results/reward_model_final.ckpt"
    acid_test_goal_distance(EXPERT_PATH, CHECKPOINT_PATH)