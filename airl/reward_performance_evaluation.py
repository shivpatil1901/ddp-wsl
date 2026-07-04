import numpy as np
import tensorflow as tf
import pickle5 as pickle
import gym
import safety_gym 
from airl_safedice import AIRLDiscriminator, load_safedice_policy, collect_rollouts


def compare_policy_performance(expert_path, checkpoint_path, policy_data, obs_dim, act_dim):
    # 1. Load learned reward model
    discriminator = AIRLDiscriminator(obs_dim, act_dim)
    # Build variables
    dummy_input = tf.zeros((1, obs_dim + act_dim))
    discriminator.g_network(dummy_input)
    discriminator.load_weights(checkpoint_path)

    def get_learned_returns(states, actions, dones, batch_size=5000):
            # Convert to numpy for slicing
            states_np = states.numpy() if hasattr(states, 'numpy') else states
            actions_np = actions.numpy() if hasattr(actions, 'numpy') else actions
            
            all_step_rewards = []
            
            # Process in batches to prevent OOM
            for i in range(0, len(states_np), batch_size):
                s_batch = tf.constant(states_np[i:i+batch_size], dtype=tf.float32)
                a_batch = tf.constant(actions_np[i:i+batch_size], dtype=tf.float32)
                sa_batch = tf.concat([s_batch, a_batch], axis=-1)
                
                # Predict rewards for this batch
                batch_rewards = discriminator.g_network(sa_batch).numpy().flatten()
                all_step_rewards.extend(batch_rewards)
            
            # Now calculate episodic returns
            episode_returns = []
            current_return = 0
            for r, done in zip(all_step_rewards, dones):
                current_return += r
                if done:
                    episode_returns.append(current_return)
                    current_return = 0
            return episode_returns

    # 2. Process Expert Data
    with open(expert_path, 'rb') as f:  
        expert_demo = pickle.load(f)
    
    expert_returns = get_learned_returns(
        tf.constant(expert_demo['states'], dtype=tf.float32),
        tf.constant(expert_demo['actions'], dtype=tf.float32),
        expert_demo['dones']
    )

    # 3. Process SafeDICE Data (from your training code's policy_data)
    policy_returns = get_learned_returns(
        tf.constant(policy_data['states'], dtype=tf.float32),
        tf.constant(policy_data['actions'], dtype=tf.float32),
        policy_data['dones']
    )

    # 4. Results
    print("\n" + "="*40)
    print(f"{'Source':<15} | {'Avg Learned Return':<20}")
    print("-" * 40)
    print(f"{'Expert':<15} | {np.mean(expert_returns):<20.4f}")
    print(f"{'SafeDICE':<15} | {np.mean(policy_returns):<20.4f}")
    print("="*40)

    if np.mean(expert_returns) > np.mean(policy_returns):
        print("✅ SUCCESS: Learned reward ranks Expert higher than SafeDICE.")
    else:
        print("⚠️ WARNING: SafeDICE has higher learned reward. Reward may be biased.")

# Usage (assuming policy_data is available in your current workspace)

EXPERT_PATH = "../SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
CHECKPOINT_PATH = "./airl_results/reward_model_final.ckpt"
CHECKPOINT_PATH_DICE = "../SafeDICE/weights/antidice_PointGoal1_seed0_20260206_184622_iter1000000.pickle"

env = gym.make('Safexp-PointGoal1-v0')
frozen_policy = load_safedice_policy(CHECKPOINT_PATH_DICE, 60, 2)

print("Running simulations to collect SafeDICE rollouts...")
policy_data = collect_rollouts(env, frozen_policy, num_episodes=10)

compare_policy_performance(EXPERT_PATH, CHECKPOINT_PATH, policy_data, 60, 2)