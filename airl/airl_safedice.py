"""
AIRL (Adversarial Inverse Reinforcement Learning) with frozen SafeDICE policy.
This script trains ONLY the discriminator to derive the reward function g(s, a).
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

# Add parent directory to path to locate SafeDICE modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SafeDICE'))

import gym
import safety_gym
from algorithms.safedice import SafeDICE as AntiDICE

# --- Model Definition ---

class AIRLDiscriminator(tf.keras.Model):
    """
    Learns f(s, a, s') = g(s, a) + γh(s') - h(s)
    g(s, a) recovers the ground truth reward.
    h(s) is the shaping term to handle transition dynamics.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, gamma=0.99):
        super(AIRLDiscriminator, self).__init__()
        self.gamma = gamma
        
        # g(s, a) - Reward Network
        self.g_network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # h(s) - Shaping Network
        self.h_network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
    def call(self, states, actions, next_states):
        sa = tf.concat([states, actions], axis=-1)
        g_sa = self.g_network(sa)
        h_s = self.h_network(states)
        h_s_next = self.h_network(next_states)
        return g_sa + self.gamma * h_s_next - h_s

    def discriminator_output(self, states, actions, next_states, log_probs):
        """
        D(s,a) = sigmoid(f(s,a,s') - log_pi(a|s))
        Numerical stability is handled by sigmoid_cross_entropy_with_logits later.
        """
        f = self.call(states, actions, next_states)
        if len(log_probs.shape) == 1:
            log_probs = tf.expand_dims(log_probs, -1)
        logits = f - log_probs
        return tf.nn.sigmoid(logits), logits

# --- Helper Functions ---

def load_safedice_policy(checkpoint_path, observation_dim, action_dim):
    config = {
        'hidden_size': 256, 'critic_lr': 3e-4, 'actor_lr': 3e-4,
        'grad_reg_coeffs': [10.0, 10.0], 'gamma': 0.99, 'alpha': 0,
        'use_last_layer_bias_cost': True, 'use_last_layer_bias_critic': True,
        'kernel_initializer': 'glorot_uniform',
    }
    policy = AntiDICE(observation_dim, action_dim, mixture_actor=False, 
                      is_discrete_action=False, config=config)
    policy.load(checkpoint_path)
    return policy

def collect_rollouts(env, policy, num_episodes=10, max_timesteps=1000):
    states, actions, next_states, log_probs, dones = [], [], [], [], []
    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(max_timesteps):
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            # SafeDICE step
            action = policy.step(state_tensor, deterministic=False).numpy()[0]
            log_p = policy.actor.get_log_prob(state_tensor, tf.convert_to_tensor([action], dtype=tf.float32))
            
            next_state, _, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            log_probs.append(log_p.numpy()[0])
            dones.append(done)
            
            state = next_state
            if done: break
    return {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'next_states': np.array(next_states, dtype=np.float32),
        'log_probs': np.array(log_probs, dtype=np.float32),
        'dones': np.array(dones, dtype=np.bool) # Add this line
    }

# --- Training Step ---

def train_step(discriminator, optimizer, expert_data, policy_data, frozen_policy, batch_size):
    # Sample data
    e_idx = np.random.randint(0, len(expert_data['states']), batch_size)
    p_idx = np.random.randint(0, len(policy_data['states']), batch_size)

    e_s, e_a, e_ns = [tf.constant(expert_data[k][e_idx]) for k in ['states', 'actions', 'next_states']]
    p_s, p_a, p_ns = [tf.constant(policy_data[k][p_idx]) for k in ['states', 'actions', 'next_states']]

    # Evaluate expert actions under the frozen policy (required for AIRL density ratio)
    # We use tf.stop_gradient to ensure the frozen policy remains untouched
    e_log_probs = tf.stop_gradient(frozen_policy.actor.get_log_prob(e_s, e_a))
    p_log_probs = tf.stop_gradient(tf.constant(policy_data['log_probs'][p_idx], dtype=tf.float32))

    with tf.GradientTape() as tape:
        _, e_logits = discriminator.discriminator_output(e_s, e_a, e_ns, e_log_probs)
        _, p_logits = discriminator.discriminator_output(p_s, p_a, p_ns, p_log_probs)

        # Loss: Expert should be 1, Policy should be 0
        loss_e = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(e_logits), logits=e_logits))
        loss_p = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(p_logits), logits=p_logits))
        total_loss = loss_e + loss_p

    # Apply gradients ONLY to discriminator variables
    grads = tape.gradient(total_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return {
        'loss': total_loss.numpy(),
        'expert_acc': tf.reduce_mean(tf.cast(e_logits > 0, tf.float32)).numpy(),
        'policy_acc': tf.reduce_mean(tf.cast(p_logits < 0, tf.float32)).numpy()
    }

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--robot_name', type=str, default='Point')
    parser.add_argument('--task_name', type=str, default='Goal1')
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    # Load Expert Demos
    expert_path = f'../SafeDICE/dataset/safetygym/ppo_lagrangian_{args.robot_name}{args.task_name}_s0.pickle'
    with open(expert_path, 'rb') as f:
        try:
            import pickle5 as pickle
        except ImportError:
            import pickle
        expert_demo = pickle.load(f)
    
    expert_data = {k: expert_demo[k][:1000 * 1000] for k in ['states', 'actions', 'next_states']}
    
    # Load Frozen Policy
    obs_dim, act_dim = expert_data['states'].shape[-1], expert_data['actions'].shape[-1]
    frozen_policy = load_safedice_policy(args.checkpoint_path, obs_dim, act_dim)

    # Env and initial data
    env = gym.make(f'Safexp-{args.robot_name}{args.task_name}-v0')
    policy_data = collect_rollouts(env, frozen_policy, num_episodes=50)

    # Discriminator and Optimizer
    discriminator = AIRLDiscriminator(obs_dim, act_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Logging setup
    os.makedirs('./airl_results', exist_ok=True)
    
    print("\n🚀 Starting AIRL training (Policy is FROZEN)...")
    for i in range(args.num_iterations):
        # Periodically refresh policy buffer if desired
        if i % 2000 == 0 and i > 0:
            new_data = collect_rollouts(env, frozen_policy, num_episodes=10)
            for k in policy_data:
                policy_data[k] = np.concatenate([policy_data[k][len(new_data[k]):], new_data[k]])

        metrics = train_step(discriminator, optimizer, expert_data, policy_data, frozen_policy, args.batch_size)

        if i % 500 == 0:
            print(f"Iter {i:5d} | Loss: {metrics['loss']:.4f} | Exp Acc: {metrics['expert_acc']:.2f} | Pol Acc: {metrics['policy_acc']:.2f}")

    # Save final learned reward
    discriminator.save_weights('./airl_results/reward_model_final.ckpt')
    print("\n Training Complete. Learned reward saved to ./airl_results/")

if __name__ == '__main__':
    main()