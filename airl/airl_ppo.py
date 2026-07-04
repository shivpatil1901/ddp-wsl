"""
Full AIRL (Adversarial Inverse Reinforcement Learning) with PPO generator.
Generator (policy) initialized from SafeDICE and updated with PPO.
Discriminator learns reward function.

Alternating training:
1. Discriminator: Learn to distinguish expert from policy
2. Generator (PPO): Update policy using learned AIRL reward

Reference: https://github.com/toshikwa/gail-airl-ppo.pytorch
"""

import numpy as np
import tensorflow as tf
import pickle
import argparse
import os
import sys
import time
import csv
from tqdm import tqdm
# from typer import clear

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SafeDICE'))

import gym
import safety_gym
from algorithms.safedice import SafeDICE as AntiDICE


class AIRLDiscriminator(tf.keras.Model):
    """AIRL discriminator that learns the reward function."""
    def __init__(self, state_dim, action_dim, hidden_size=256, gamma=0.99):
        super(AIRLDiscriminator, self).__init__()
        self.gamma = gamma
        
        self.g_network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        self.h_network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
    def call(self, states, actions, next_states):
        """Compute f(s, a, s') = g(s, a) + γh(s') - h(s)"""
        sa = tf.concat([states, actions], axis=-1)
        g_sa = self.g_network(sa)
        h_s = self.h_network(states)
        h_s_next = self.h_network(next_states)
        f = g_sa + self.gamma * h_s_next - h_s
        return f
    
    def get_reward(self, states, actions, next_states, log_probs):
        """Get learned reward: r(s, a, s') = f(s, a, s') - log π(a|s)"""
        f = self.call(states, actions, next_states)
        reward = f - log_probs
        return reward
    
    def discriminator_output(self, states, actions, next_states, log_probs):
        """Compute discriminator output for training"""
        f = self.call(states, actions, next_states)
        logits = f - log_probs
        return tf.nn.sigmoid(logits), logits


class PPOBuffer:
    """Buffer for storing trajectories for PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.log_probs = []
        self.rewards = []  # AIRL rewards
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def store(self, state, action, next_state, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def finish_path(self, last_value=0):
        """Compute GAE advantages and returns"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalue = last_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalue = values[t + 1]
            delta = rewards[t] + 0.99 * nextvalue * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
        
        returns = advantages + values
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get(self, clear=True):
        """Get all data and optionally clear buffer"""
        data = dict(
            states=np.array(self.states, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32),
            next_states=np.array(self.next_states, dtype=np.float32),
            log_probs=np.array(self.log_probs, dtype=np.float32),
            advantages=np.array(self.advantages, dtype=np.float32),
            returns=np.array(self.returns, dtype=np.float32),
            values=np.array(self.values, dtype=np.float32)
        )
        if clear:
            self.states, self.actions, self.next_states = [], [], []
            self.log_probs, self.rewards, self.values = [], [], []
            self.dones, self.advantages, self.returns = [], [], []
        return data


def load_safedice_policy(checkpoint_path, observation_dim, action_dim):
    """Load trained SafeDICE policy as initialization."""
    print(f"Loading SafeDICE policy from: {checkpoint_path}")
    
    config = {
        'hidden_size': 256,
        'critic_lr': 3e-4,
        'actor_lr': 3e-4,
        'grad_reg_coeffs': [10.0, 10.0],
        'gamma': 0.99,
        'alpha': 0,
        'use_last_layer_bias_cost': True,
        'use_last_layer_bias_critic': True,
        'kernel_initializer': 'glorot_uniform',
    }
    
    policy = AntiDICE(observation_dim, action_dim, mixture_actor=False, 
                      is_discrete_action=False, config=config)
    policy.load(checkpoint_path)
    
    print("✅ SafeDICE policy loaded as initialization")
    return policy


def collect_trajectories_for_ppo(env, policy, discriminator, buffer, num_steps=2048):
    """Collect trajectories for PPO update using AIRL learned reward."""
    state = env.reset()
    episode_reward = 0
    episode_cost = 0
    episode_length = 0
    
    episodes_rewards = []
    episodes_costs = []
    episodes_lengths = []
    
    for step in range(num_steps):
        # Get action from policy
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action = policy.step(state_tensor, deterministic=False).numpy()[0]

        action = np.clip(action, -1.0, 1.0)
        
        # Get value estimate
        value, _ = policy.critic(state_tensor)
        value = float(value.numpy().flatten()[0])  # Fixed: handle both 1D and 2D arrays
        
        # Get log prob
        log_prob = policy.actor.get_log_prob(
            state_tensor, 
            tf.convert_to_tensor([action], dtype=tf.float32)
        ).numpy().flatten()[0]  # Fixed: flatten to get scalar
        
        # Take step in environment
        next_state, env_reward, done, info = env.step(action)
        cost = info.get('cost', 0)
        
        # Compute AIRL reward
        next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
        airl_reward = discriminator.get_reward(
            state_tensor,
            tf.convert_to_tensor([action], dtype=tf.float32),
            next_state_tensor,
            tf.convert_to_tensor([[log_prob]], dtype=tf.float32)
        ).numpy().flatten()[0]  # Fixed: flatten to get scalar
        
        # Store transition
        buffer.store(state, action, next_state, log_prob, airl_reward, value, done)
        
        episode_reward += env_reward
        episode_cost += cost
        episode_length += 1
        state = next_state
        
        if done:
            # Finish episode
            buffer.finish_path(last_value=0)
            episodes_rewards.append(episode_reward)
            episodes_costs.append(episode_cost)
            episodes_lengths.append(episode_length)
            
            state = env.reset()
            episode_reward = 0
            episode_cost = 0
            episode_length = 0
    
    # Finish any incomplete trajectory
    if episode_length > 0:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        last_value, _ = policy.critic(state_tensor)
        last_value = float(last_value.numpy().flatten()[0])  # Fixed: handle both 1D and 2D arrays
        buffer.finish_path(last_value=last_value)
        episodes_rewards.append(episode_reward)
        episodes_costs.append(episode_cost)
        episodes_lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(episodes_rewards) if episodes_rewards else 0,
        'mean_cost': np.mean(episodes_costs) if episodes_costs else 0,
        'mean_length': np.mean(episodes_lengths) if episodes_lengths else 0,
        'num_episodes': len(episodes_rewards)
    }


def update_ppo(policy, buffer, clip_ratio=0.2, target_kl=0.015, train_iters=10):  # Reduced target_kl
    """Update policy using PPO algorithm."""
    data = buffer.get(clear=False)
    
    # Check if buffer has data
    if len(data['states']) == 0:
        print("⚠️  Warning: Empty buffer, skipping PPO update")
        return {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'kl': 0.0
        }
    
    states = tf.constant(data['states'])
    actions = tf.constant(data['actions'])
    old_log_probs = tf.constant(data['log_probs'])
    advantages = tf.constant(data['advantages'])
    returns = tf.constant(data['returns'])
    
    # Normalize advantages
    advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
    
    # Clip advantages to prevent extreme updates
    advantages = tf.clip_by_value(advantages, -10.0, 10.0)
    
    dataset_size = states.shape[0]
    batch_size = min(256, dataset_size)
    
    # Ensure batch_size is at least 1
    if batch_size == 0:
        batch_size = dataset_size
    
    # If dataset is too small, just do full batch updates
    if dataset_size < 64:
        batch_size = dataset_size
    
    actor_losses = []
    critic_losses = []
    kls = []
    
    for epoch in range(train_iters):
        # Shuffle data
        indices = np.random.permutation(dataset_size)
        
        for start in range(0, dataset_size, max(1, batch_size)):
            end = min(start + batch_size, dataset_size)
            batch_indices = indices[start:end]
            
            batch_states = tf.gather(states, batch_indices)
            batch_actions = tf.gather(actions, batch_indices)
            batch_old_log_probs = tf.gather(old_log_probs, batch_indices)
            batch_advantages = tf.gather(advantages, batch_indices)
            batch_returns = tf.gather(returns, batch_indices)
            
            # Update actor
            with tf.GradientTape() as tape:
                new_log_probs = policy.actor.get_log_prob(batch_states, batch_actions)
                ratio = tf.exp(new_log_probs - batch_old_log_probs)
                
                clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
                actor_loss = -tf.reduce_mean(
                    tf.minimum(ratio * batch_advantages, 
                              clipped_ratio * batch_advantages)
                )
                
                # Entropy bonus (increased for more exploration)
                entropy = -tf.reduce_mean(new_log_probs)
                actor_loss = actor_loss - 0.05 * entropy  # Increased from 0.01
            
            actor_grads = tape.gradient(actor_loss, policy.actor.variables)
            # Clip gradients
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.5)
            policy.actor_optimizer.apply_gradients(zip(actor_grads, policy.actor.variables))
            
            # Update critic
            with tf.GradientTape() as tape:
                values, _ = policy.critic(batch_states)
                values = tf.squeeze(values)
                critic_loss = tf.reduce_mean(tf.square(values - batch_returns))
            
            critic_grads = tape.gradient(critic_loss, policy.critic.variables)
            # Clip gradients
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, 0.5)
            policy.critic_optimizer.apply_gradients(zip(critic_grads, policy.critic.variables))
            
            # Compute KL for early stopping
            kl = tf.reduce_mean(batch_old_log_probs - new_log_probs)
            
            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss.numpy())
            kls.append(kl.numpy())
        
        # Early stopping based on KL divergence
        mean_kl = np.mean(kls) if kls else 0
        if mean_kl > 1.5 * target_kl:
            print(f"  ⚠️ Early stopping at epoch {epoch+1} (KL={mean_kl:.4f} > {1.5*target_kl:.4f})")
            break
    
    return {
        'actor_loss': np.mean(actor_losses) if actor_losses else 0.0,
        'critic_loss': np.mean(critic_losses) if critic_losses else 0.0,
        'kl': np.mean(kls) if kls else 0.0
    }

def train_discriminator(discriminator, optimizer, expert_data, policy_data, policy, batch_size=256):
    """Train AIRL discriminator."""
    expert_indices = np.random.randint(0, len(expert_data['states']), size=batch_size)
    policy_indices = np.random.randint(0, len(policy_data['states']), size=batch_size)
    
    expert_states = tf.constant(expert_data['states'][expert_indices], dtype=tf.float32)
    expert_actions = tf.constant(expert_data['actions'][expert_indices], dtype=tf.float32)
    expert_next_states = tf.constant(expert_data['next_states'][expert_indices], dtype=tf.float32)
    
    policy_states = tf.constant(policy_data['states'][policy_indices], dtype=tf.float32)
    policy_actions = tf.constant(policy_data['actions'][policy_indices], dtype=tf.float32)
    policy_next_states = tf.constant(policy_data['next_states'][policy_indices], dtype=tf.float32)

    # Get log probs
    expert_log_probs = policy.actor.get_log_prob(expert_states, expert_actions)
    policy_log_probs = policy.actor.get_log_prob(policy_states, policy_actions)
    
    with tf.GradientTape() as tape:
        expert_d, expert_logits = discriminator.discriminator_output(
            expert_states, expert_actions, expert_next_states, expert_log_probs
        )
        policy_d, policy_logits = discriminator.discriminator_output(
            policy_states, policy_actions, policy_next_states, policy_log_probs
        )
        
        expert_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(expert_logits), logits=expert_logits
            )
        )
        policy_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(policy_logits), logits=policy_logits
            )
        )
        
        total_loss = expert_loss + policy_loss
        
        # Gradient penalty
        alpha = tf.random.uniform([batch_size, 1])
        interpolated_states = alpha * expert_states + (1 - alpha) * policy_states
        interpolated_actions = alpha * expert_actions + (1 - alpha) * policy_actions
        interpolated_next_states = alpha * expert_next_states + (1 - alpha) * policy_next_states
        interpolated_log_probs = alpha * expert_log_probs + (1 - alpha) * policy_log_probs
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch([interpolated_states, interpolated_actions, interpolated_next_states])
            _, interpolated_logits = discriminator.discriminator_output(
                interpolated_states, interpolated_actions, 
                interpolated_next_states, interpolated_log_probs
            )
        
        gradients = gp_tape.gradient(interpolated_logits, 
                                     [interpolated_states, interpolated_actions, interpolated_next_states])
        grad_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in gradients if g is not None]) + 1e-8)
        gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
        
        total_loss += 10.0 * gradient_penalty
    
    gradients = tape.gradient(total_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    
    expert_acc = tf.reduce_mean(tf.cast(expert_d > 0.5, tf.float32))
    policy_acc = tf.reduce_mean(tf.cast(policy_d < 0.5, tf.float32))
    
    return {
        'discriminator_loss': total_loss.numpy(),
        'expert_accuracy': expert_acc.numpy(),
        'policy_accuracy': policy_acc.numpy(),
        'expert_d_mean': tf.reduce_mean(expert_d).numpy(),
        'policy_d_mean': tf.reduce_mean(policy_d).numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--robot_name', type=str, default='Point')
    parser.add_argument('--task_name', type=str, default='Goal1')
    parser.add_argument('--num_expert_trajectories', type=int, default=1000)
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--ppo_steps_per_iter', type=int, default=2048)
    parser.add_argument('--disc_updates_per_iter', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=10)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU found: {gpus[0]}")
    
    # Load expert data
    print("Loading expert demonstrations...")
    expert_path = f'../SafeDICE/dataset/safetygym/ppo_lagrangian_{args.robot_name}{args.task_name}_s0.pickle'
    with open(expert_path, 'rb') as f:
        try:
            import pickle5 as pickle
        except ImportError:
            import pickle
        expert_demo = pickle.load(f)
    
    max_timesteps = 1000
    expert_data = {
        'states': expert_demo['states'][:args.num_expert_trajectories * max_timesteps],
        'actions': expert_demo['actions'][:args.num_expert_trajectories * max_timesteps],
        'next_states': expert_demo['next_states'][:args.num_expert_trajectories * max_timesteps],
    }
    print(f"Loaded {len(expert_data['states'])} expert transitions")
    
    # Initialize policy from SafeDICE
    observation_dim = expert_data['states'].shape[-1]
    action_dim = expert_data['actions'].shape[-1]
    policy = load_safedice_policy(args.checkpoint_path, observation_dim, action_dim)
    
    # Initialize discriminator
    discriminator = AIRLDiscriminator(observation_dim, action_dim, gamma=args.gamma)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # Create environment
    env_name = f'Safexp-{args.robot_name}{args.task_name}-v0'
    env = gym.make(env_name)
    print(f"Environment: {env_name}")

    # ========== PRE-TRAIN DISCRIMINATOR ==========
    print("\ Pre-training discriminator (1000 iterations)...")
    
    # Collect initial policy rollouts from SafeDICE
    initial_buffer = PPOBuffer()
    temp_disc = AIRLDiscriminator(observation_dim, action_dim, gamma=args.gamma)  # Dummy for collection
    _ = collect_trajectories_for_ppo(env, policy, temp_disc, initial_buffer, num_steps=10000)
    initial_data = initial_buffer.get()
    
    policy_rollout_data = {
        'states': initial_data['states'],
        'actions': initial_data['actions'],
        'next_states': initial_data['next_states']
    }
    
    for pretrain_iter in range(1000):
        disc_stats = train_discriminator(
            discriminator, disc_optimizer, expert_data,
            policy_rollout_data, policy, batch_size=args.batch_size
        )
        if pretrain_iter % 10 == 0:
            print(f"  Iter {pretrain_iter}: Expert={disc_stats['expert_accuracy']:.3f} "
                  f"Policy={disc_stats['policy_accuracy']:.3f} "
                  f"Loss={disc_stats['discriminator_loss']:.3f}")
    
    print("✅ Discriminator pre-training complete\n")
    # ========== END PRE-TRAIN ==========
    
    # Setup logging
    log_dir = './airl_ppo_logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"{log_dir}/airl_ppo_{args.robot_name}{args.task_name}_seed{args.seed}_{timestamp}.csv"
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'iteration', 'mean_reward', 'mean_cost', 'mean_length',
        'actor_loss', 'critic_loss', 'kl', 'disc_loss',
        'expert_acc', 'policy_acc', 'expert_d', 'policy_d'
    ])
    csv_file.flush()
    print(f"Logging to: {csv_path}")
    
    # Training loop
    print("\nStarting AIRL-PPO training...")
    start_time = time.time()
    
    # Buffer for storing policy rollouts (for discriminator training)
    policy_rollout_buffer = {'states': [], 'actions': [], 'next_states': []}
    
    with tqdm(total=args.num_iterations, desc='AIRL-PPO') as pbar:
        for iteration in range(args.num_iterations):
            # Collect trajectories and update policy with PPO
            ppo_buffer = PPOBuffer()
            rollout_stats = collect_trajectories_for_ppo(
                env, policy, discriminator, ppo_buffer, 
                num_steps=args.ppo_steps_per_iter
            )
            
            # Get policy data for discriminator
            ppo_data = ppo_buffer.get(clear=False)
            policy_rollout_buffer['states'] = ppo_data['states'][-10000:].tolist()
            policy_rollout_buffer['actions'] = ppo_data['actions'][-10000:].tolist()
            policy_rollout_buffer['next_states'] = ppo_data['next_states'][-10000:].tolist()
            
            # Update policy with PPO
            ppo_stats = update_ppo(policy, ppo_buffer)
            
            # Update discriminator multiple times
            disc_stats_list = []
            policy_rollout_data = {
                'states': np.array(policy_rollout_buffer['states']),
                'actions': np.array(policy_rollout_buffer['actions']),
                'next_states': np.array(policy_rollout_buffer['next_states'])
            }
            
            for _ in range(args.disc_updates_per_iter):
                disc_stats = train_discriminator(
                    discriminator, disc_optimizer, expert_data,
                    policy_rollout_data, policy, batch_size=args.batch_size
                )
                disc_stats_list.append(disc_stats)
            
            # Average discriminator stats
            disc_stats_avg = {
                key: np.mean([stats[key] for stats in disc_stats_list])
                for key in disc_stats_list[0].keys()
            }
            
            # Log metrics
            if iteration % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"\nIteration {iteration} ({elapsed:.1f}s)")
                print(f"  Reward: {rollout_stats['mean_reward']:.2f} | Cost: {rollout_stats['mean_cost']:.2f}")
                print(f"  Actor Loss: {ppo_stats['actor_loss']:.4f} | KL: {ppo_stats['kl']:.4f}")
                print(f"  Disc: Expert={disc_stats_avg['expert_accuracy']:.3f} Policy={disc_stats_avg['policy_accuracy']:.3f}")
                
                csv_writer.writerow([
                    iteration,
                    rollout_stats['mean_reward'],
                    rollout_stats['mean_cost'],
                    rollout_stats['mean_length'],
                    ppo_stats['actor_loss'],
                    ppo_stats['critic_loss'],
                    ppo_stats['kl'],
                    disc_stats_avg['discriminator_loss'],
                    disc_stats_avg['expert_accuracy'],
                    disc_stats_avg['policy_accuracy'],
                    disc_stats_avg['expert_d_mean'],
                    disc_stats_avg['policy_d_mean']
                ])
                csv_file.flush()
            
            pbar.update(1)
    
    # Save final models
    checkpoint_dir = './airl_ppo_weights'
    os.makedirs(checkpoint_dir, exist_ok=True)
    policy_path = f"{checkpoint_dir}/policy_{args.robot_name}{args.task_name}_final.pickle"
    disc_path = f"{checkpoint_dir}/disc_{args.robot_name}{args.task_name}_final.pkl"
    
    training_info = {'iteration': args.num_iterations, 'logs': []}
    policy.save(policy_path, training_info)
    discriminator.save_weights(disc_path)
    
    print(f"\n✅ Training complete!")
    print(f"Policy saved to: {policy_path}")
    print(f"Discriminator saved to: {disc_path}")
    
    csv_file.close()
    env.close()


if __name__ == '__main__':
    main()
