import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym
import safety_gym
import scipy.stats as stats
from airl_safedice import AIRLDiscriminator

# def collect_new_rollouts(env, discriminator, num_episodes=10, noise_std=0.4):
#     """
#     Generates new data by taking random actions to ensure 
#     high exploration and hazard collisions.
#     """
#     obs_list, act_list, cost_list = [], [], []
    
#     for ep in range(num_episodes):
#         obs = env.reset()
#         done = False
#         while not done:
#             # Random sampling ensures we hit hazards that the expert avoids
#             action = env.action_space.sample() 
#             next_obs, reward, done, info = env.step(action)
            
#             obs_list.append(obs)
#             act_list.append(action)
#             cost_list.append(info.get('cost', 0))
            
#             obs = next_obs
            
#     return np.array(obs_list), np.array(act_list), np.array(cost_list)


def collect_creep_rollouts(env, discriminator, num_episodes=5, creep_speed=0.1):
    obs_list, act_list, cost_list = [], [], []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            # 1. Identify where the hazards are using Lidar (Indices 16-31)
            hazard_lidar = obs[16:32]
            target_bin = np.argmax(hazard_lidar)
            
            # 2. Simple logic: Turn toward the strongest hazard signal and move SLOWLY
            # This is a 'dumb' creep to force the agent into the hazard boundary
            action = np.zeros(env.action_space.shape)
            
            # Action[0] is typically drive, Action[1] is turn in Point agents
            action[0] = creep_speed  # Move forward slowly
            # Turn toward the Lidar bin with the highest reading
            action[1] = (target_bin - 8) / 8.0 
            
            next_obs, reward, done, info = env.step(action)
            
            obs_list.append(obs)
            act_list.append(action)
            cost_list.append(info.get('cost', 0))
            
            obs = next_obs
            
    return np.array(obs_list), np.array(act_list), np.array(cost_list)

def run_advanced_safety_acid_test(checkpoint_path, obs_dim=60, act_dim=2):
    # 1. Initialize Environment
    env = gym.make('Safexp-PointGoal1-v0')
    
    # 2. Load and Build Discriminator
    discriminator = AIRLDiscriminator(obs_dim, act_dim)
    _ = discriminator.g_network(tf.zeros((1, obs_dim + act_dim)))
    discriminator.load_weights(checkpoint_path)
    print(f"Model loaded from {checkpoint_path}")

    # 3. Collect New Rollout Data
    print("Collecting new noisy rollouts for hazard testing...")
    obs_data, act_data, cost_data = collect_creep_rollouts(env, discriminator)

    # 4. Compute Learned Rewards g(s, a)
    sa_input = tf.concat([tf.cast(obs_data, tf.float32), tf.cast(act_data, tf.float32)], axis=-1)
    learned_rewards = discriminator.g_network(sa_input).numpy().flatten()

    # 5. COST NATURE DIAGNOSTIC
    unique_costs = np.sort(np.unique(cost_data))
    print("-" * 30)
    print("COST NATURE DIAGNOSTIC")
    print("-" * 30)
    print(f"Total unique cost values detected: {len(unique_costs)}")

    if len(unique_costs) > 2:
        print("Result: The cost is CONTINUOUS (Gradient-based).")
    else:
        print("Result: The cost appears BINARY/DISCRETE in this data.")

    print("\nSample of Unique Cost Values:")
    sample_size = min(5, len(unique_costs))
    for i in range(sample_size):
        print(f"  Value {i+1}: {unique_costs[i]:.6f}")
    
    if len(unique_costs) == 0:
        print("  [No cost values found]")
    elif len(unique_costs) < 5:
        print(f"  [Only {len(unique_costs)} unique values available]")
    print("-" * 30)

    # 6. Slicing Observations (Hazards: 16-31, Vases: 32-47)
    max_hazard_signal = np.max(obs_data[:, 16:32], axis=1)
    max_vase_signal = np.max(obs_data[:, 32:48], axis=1)

    # 7. Calculate Spearman Correlations
    haz_corr, _ = stats.spearmanr(max_hazard_signal, learned_rewards)
    vas_corr, _ = stats.spearmanr(max_vase_signal, learned_rewards)

    # 8. Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Hazard Sensitivity
    axs[0].scatter(max_hazard_signal, learned_rewards, alpha=0.3, color='crimson')
    axs[0].set_title(f"Hazard Lidar vs. g(s,a)\nSpearman r: {haz_corr:.3f}")
    axs[0].set_xlabel("Max Hazard Lidar Signal")
    axs[0].set_ylabel("Learned Reward")

    # Plot 2: Vase Neutrality (Sanity Check)
    axs[1].scatter(max_vase_signal, learned_rewards, alpha=0.3, color='gray')
    axs[1].set_title(f"Vase Lidar vs. g(s,a)\nSpearman r: {vas_corr:.3f}")
    axs[1].set_xlabel("Max Vase Lidar Signal")

    # Plot 3: Cost Correlation
    active_idx = cost_data > 0
    if np.any(active_idx):
        axs[2].scatter(cost_data[active_idx], learned_rewards[active_idx], color='purple', alpha=0.5)
        axs[2].set_title("Active Env Cost vs. g(s,a)")
        axs[2].set_xlabel("Cost Value")
    else:
        axs[2].text(0.5, 0.5, "No Active Costs Found", ha='center')

    plt.tight_layout()
    plt.savefig("safety_acid_test_refined.png")
    print(f"\nPlot saved as safety_acid_test_refined.png")
    print(f"Hazard Correlation: {haz_corr:.3f}, Vase Correlation: {vas_corr:.3f}")

if __name__ == "__main__":
    CHECKPOINT_PATH = "./airl_results/reward_model_final.ckpt"
    # Ensure obs_dim and act_dim match your saved model
    run_advanced_safety_acid_test(CHECKPOINT_PATH, obs_dim=60, act_dim=2)