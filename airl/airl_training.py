"""
Standard AIRL with PPO generator updates.

This script alternates:
1) Discriminator (reward) updates using expert vs policy transitions.
2) PPO updates of the generator policy using AIRL reward.

Outputs:
- Reloadable policy checkpoint (.pickle via SafeDICE policy.save)
- Reloadable reward checkpoint (.ckpt via TF save_weights)
- Training metrics CSV and run metadata
"""

import argparse
import csv
import json
import os
try:
    import pickle5 as pickle
except ImportError:
    import pickle
import sys
import time

import gym
import numpy as np
import safety_gym
import tensorflow as tf

# Allow importing SafeDICE policy implementation.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SafeDICE'))
from algorithms.safedice import SafeDICE as AntiDICE


def safe_reset(env):
	out = env.reset()
	if isinstance(out, tuple):
		return out[0]
	return out


def safe_step(env, action):
	out = env.step(action)
	if len(out) == 5:
		next_state, reward, terminated, truncated, info = out
		done = terminated or truncated
		return next_state, reward, done, info
	return out


def discounted_cumsum(x, discount):
	out = np.zeros_like(x, dtype=np.float32)
	running = 0.0
	for i in reversed(range(len(x))):
		running = x[i] + discount * running
		out[i] = running
	return out


class AIRLDiscriminator(tf.keras.Model):
	"""AIRL reward model with potential shaping."""

	def __init__(self, state_dim, action_dim, hidden_size=256, gamma=0.99):
		super(AIRLDiscriminator, self).__init__()
		self.gamma = gamma
		self.g_network = tf.keras.Sequential([
			tf.keras.layers.Dense(hidden_size, activation='relu'),
			tf.keras.layers.Dense(hidden_size, activation='relu'),
			tf.keras.layers.Dense(1),
		])
		self.h_network = tf.keras.Sequential([
			tf.keras.layers.Dense(hidden_size, activation='relu'),
			tf.keras.layers.Dense(hidden_size, activation='relu'),
			tf.keras.layers.Dense(1),
		])

	def f(self, states, actions, next_states):
		sa = tf.concat([states, actions], axis=-1)
		g_sa = self.g_network(sa)
		h_s = self.h_network(states)
		h_next = self.h_network(next_states)
		return g_sa + self.gamma * h_next - h_s

	def logits(self, states, actions, next_states, log_probs):
		log_probs = tf.reshape(log_probs, (-1, 1))
		return self.f(states, actions, next_states) - log_probs

	def shaped_reward(self, states, actions, next_states, log_probs):
		return self.logits(states, actions, next_states, log_probs)

	def transfer_reward(self, states, actions):
		"""Reward for transfer studies, independent of current policy log-prob."""
		sa = tf.concat([states, actions], axis=-1)
		return self.g_network(sa)


class RolloutBuffer(object):
	"""Trajectory buffer with GAE-Lambda advantages for PPO."""

	def __init__(self, gamma=0.99, lam=0.95):
		self.gamma = gamma
		self.lam = lam
		self.clear()

	def clear(self):
		self.states = []
		self.actions = []
		self.next_states = []
		self.log_probs = []
		self.rewards = []
		self.values = []
		self.advantages = []
		self.returns = []
		self.path_start_idx = 0

	def store(self, state, action, next_state, log_prob, reward, value):
		self.states.append(state)
		self.actions.append(action)
		self.next_states.append(next_state)
		self.log_probs.append(log_prob)
		self.rewards.append(reward)
		self.values.append(value)

	def finish_path(self, last_value=0.0):
		path_slice = slice(self.path_start_idx, len(self.rewards))
		rewards = np.array(self.rewards[path_slice] + [last_value], dtype=np.float32)
		values = np.array(self.values[path_slice] + [last_value], dtype=np.float32)

		deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
		advantages = discounted_cumsum(deltas, self.gamma * self.lam)
		returns = discounted_cumsum(rewards, self.gamma)[:-1]

		self.advantages.extend(advantages.tolist())
		self.returns.extend(returns.tolist())
		self.path_start_idx = len(self.rewards)

	def get(self):
		data = {
			'states': np.array(self.states, dtype=np.float32),
			'actions': np.array(self.actions, dtype=np.float32),
			'next_states': np.array(self.next_states, dtype=np.float32),
			'log_probs': np.array(self.log_probs, dtype=np.float32),
			'advantages': np.array(self.advantages, dtype=np.float32),
			'returns': np.array(self.returns, dtype=np.float32),
			'values': np.array(self.values, dtype=np.float32),
		}
		return data


def make_policy(obs_dim, act_dim, init_policy_path=None):
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
	policy = AntiDICE(obs_dim, act_dim, mixture_actor=False, is_discrete_action=False, config=config)
	if init_policy_path:
		print('Loading initial policy from: %s' % init_policy_path)
		policy.load(init_policy_path)
	return policy


def policy_action_value_logp(policy, state, deterministic=False):
	state_t = tf.convert_to_tensor([state], dtype=tf.float32)
	action = policy.step(state_t, deterministic=deterministic).numpy()[0]

	action_t = tf.convert_to_tensor([action], dtype=tf.float32)
	log_prob_t = policy.actor.get_log_prob(state_t, action_t)
	value_t, _ = policy.critic(state_t)

	log_prob = float(np.squeeze(log_prob_t.numpy()))
	value = float(np.squeeze(value_t.numpy()))
	return action, value, log_prob


def collect_rollout(env, policy, discriminator, buffer, steps_per_iter, max_ep_len):
	state = safe_reset(env)
	ep_reward = 0.0
	ep_cost = 0.0
	ep_len = 0

	ep_rewards = []
	ep_costs = []
	ep_lengths = []

	for t in range(steps_per_iter):
		action, value, log_prob = policy_action_value_logp(policy, state, deterministic=False)
		action = np.clip(action, env.action_space.low, env.action_space.high)

		next_state, true_reward, done, info = safe_step(env, action)
		cost = float(info.get('cost', 0.0))

		state_t = tf.convert_to_tensor([state], dtype=tf.float32)
		action_t = tf.convert_to_tensor([action], dtype=tf.float32)
		next_state_t = tf.convert_to_tensor([next_state], dtype=tf.float32)
		log_prob_t = tf.convert_to_tensor([[log_prob]], dtype=tf.float32)

		airl_reward = float(np.squeeze(discriminator.shaped_reward(state_t, action_t, next_state_t, log_prob_t).numpy()))
		buffer.store(state, action, next_state, log_prob, airl_reward, value)

		ep_reward += float(true_reward)
		ep_cost += cost
		ep_len += 1
		state = next_state

		timeout = (ep_len >= max_ep_len)
		terminal = done or timeout
		epoch_ended = (t == steps_per_iter - 1)

		if terminal or epoch_ended:
			if terminal and not done:
				_, last_value, _ = policy_action_value_logp(policy, state, deterministic=False)
			elif epoch_ended and not terminal:
				_, last_value, _ = policy_action_value_logp(policy, state, deterministic=False)
			else:
				last_value = 0.0

			buffer.finish_path(last_value=last_value)

			if terminal:
				ep_rewards.append(ep_reward)
				ep_costs.append(ep_cost)
				ep_lengths.append(ep_len)
				state = safe_reset(env)
				ep_reward = 0.0
				ep_cost = 0.0
				ep_len = 0

	return {
		'mean_ep_reward': float(np.mean(ep_rewards)) if ep_rewards else 0.0,
		'mean_ep_cost': float(np.mean(ep_costs)) if ep_costs else 0.0,
		'mean_ep_length': float(np.mean(ep_lengths)) if ep_lengths else 0.0,
		'episodes': len(ep_rewards),
	}


def ppo_update(policy, data, clip_ratio, target_kl, train_pi_iters, train_v_iters, batch_size, entropy_coef):
	states = tf.convert_to_tensor(data['states'], dtype=tf.float32)
	actions = tf.convert_to_tensor(data['actions'], dtype=tf.float32)
	old_logp = tf.convert_to_tensor(data['log_probs'], dtype=tf.float32)
	returns = tf.convert_to_tensor(data['returns'], dtype=tf.float32)

	advantages = data['advantages']
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
	advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

	n = states.shape[0]
	if n == 0:
		return {'pi_loss': 0.0, 'v_loss': 0.0, 'kl': 0.0, 'entropy': 0.0}

	batch_size = int(min(max(1, batch_size), n))

	pi_losses = []
	v_losses = []
	kls = []
	entropies = []

	# Policy optimization
	for _ in range(train_pi_iters):
		perm = np.random.permutation(n)
		stop_early = False

		for start in range(0, n, batch_size):
			idx = perm[start:start + batch_size]
			b_states = tf.gather(states, idx)
			b_actions = tf.gather(actions, idx)
			b_old_logp = tf.gather(old_logp, idx)
			b_adv = tf.gather(advantages, idx)

			with tf.GradientTape() as tape:
				new_logp = policy.actor.get_log_prob(b_states, b_actions)
				new_logp = tf.reshape(new_logp, (-1,))
				ratio = tf.exp(new_logp - b_old_logp)

				clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
				pi_loss = -tf.reduce_mean(tf.minimum(ratio * b_adv, clipped * b_adv))

				entropy = -tf.reduce_mean(new_logp)
				pi_loss = pi_loss - entropy_coef * entropy

			grads = tape.gradient(pi_loss, policy.actor.variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.5)
			policy.actor_optimizer.apply_gradients(zip(grads, policy.actor.variables))

			approx_kl = tf.reduce_mean(b_old_logp - new_logp)

			pi_losses.append(float(pi_loss.numpy()))
			kls.append(float(approx_kl.numpy()))
			entropies.append(float(entropy.numpy()))

			if float(approx_kl.numpy()) > 1.5 * target_kl:
				stop_early = True
				break

		if stop_early:
			break

	# Value function optimization
	for _ in range(train_v_iters):
		perm = np.random.permutation(n)
		for start in range(0, n, batch_size):
			idx = perm[start:start + batch_size]
			b_states = tf.gather(states, idx)
			b_returns = tf.gather(returns, idx)

			with tf.GradientTape() as tape:
				values, _ = policy.critic(b_states)
				values = tf.reshape(values, (-1,))
				v_loss = tf.reduce_mean(tf.square(values - b_returns))

			grads = tape.gradient(v_loss, policy.critic.variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.5)
			policy.critic_optimizer.apply_gradients(zip(grads, policy.critic.variables))
			v_losses.append(float(v_loss.numpy()))

	return {
		'pi_loss': float(np.mean(pi_losses)) if pi_losses else 0.0,
		'v_loss': float(np.mean(v_losses)) if v_losses else 0.0,
		'kl': float(np.mean(kls)) if kls else 0.0,
		'entropy': float(np.mean(entropies)) if entropies else 0.0,
	}


def discriminator_update(discriminator, optimizer, expert_data, policy_data, policy, batch_size):
	e_n = expert_data['states'].shape[0]
	p_n = policy_data['states'].shape[0]
	if e_n == 0 or p_n == 0:
		return {
			'disc_loss': 0.0,
			'expert_acc': 0.0,
			'policy_acc': 0.0,
			'expert_prob': 0.0,
			'policy_prob': 0.0,
		}

	e_idx = np.random.randint(0, e_n, size=batch_size)
	p_idx = np.random.randint(0, p_n, size=batch_size)

	e_s = tf.convert_to_tensor(expert_data['states'][e_idx], dtype=tf.float32)
	e_a = tf.convert_to_tensor(expert_data['actions'][e_idx], dtype=tf.float32)
	e_ns = tf.convert_to_tensor(expert_data['next_states'][e_idx], dtype=tf.float32)

	p_s = tf.convert_to_tensor(policy_data['states'][p_idx], dtype=tf.float32)
	p_a = tf.convert_to_tensor(policy_data['actions'][p_idx], dtype=tf.float32)
	p_ns = tf.convert_to_tensor(policy_data['next_states'][p_idx], dtype=tf.float32)

	e_logp = policy.actor.get_log_prob(e_s, e_a)
	p_logp = policy.actor.get_log_prob(p_s, p_a)

	with tf.GradientTape() as tape:
		e_logits = discriminator.logits(e_s, e_a, e_ns, e_logp)
		p_logits = discriminator.logits(p_s, p_a, p_ns, p_logp)

		e_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=tf.ones_like(e_logits), logits=e_logits
		))
		p_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			labels=tf.zeros_like(p_logits), logits=p_logits
		))
		loss = e_loss + p_loss

	grads = tape.gradient(loss, discriminator.trainable_variables)
	optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

	e_prob = tf.nn.sigmoid(e_logits)
	p_prob = tf.nn.sigmoid(p_logits)
	expert_acc = tf.reduce_mean(tf.cast(e_prob > 0.5, tf.float32))
	policy_acc = tf.reduce_mean(tf.cast(p_prob < 0.5, tf.float32))

	return {
		'disc_loss': float(loss.numpy()),
		'expert_acc': float(expert_acc.numpy()),
		'policy_acc': float(policy_acc.numpy()),
		'expert_prob': float(tf.reduce_mean(e_prob).numpy()),
		'policy_prob': float(tf.reduce_mean(p_prob).numpy()),
	}


def load_expert_data(expert_path, max_steps=None):
	with open(expert_path, 'rb') as f:
		data = pickle.load(f)

	required = ['states', 'actions', 'next_states']
	for key in required:
		if key not in data:
			raise ValueError('Expert file missing key: %s' % key)

	states = np.asarray(data['states'], dtype=np.float32)
	actions = np.asarray(data['actions'], dtype=np.float32)
	next_states = np.asarray(data['next_states'], dtype=np.float32)

	n = min(len(states), len(actions), len(next_states))
	if max_steps is not None and max_steps > 0:
		n = min(n, int(max_steps))

	return {
		'states': states[:n],
		'actions': actions[:n],
		'next_states': next_states[:n],
	}


def ensure_discriminator_built(discriminator, obs_dim, act_dim):
	s = tf.zeros((1, obs_dim), dtype=tf.float32)
	a = tf.zeros((1, act_dim), dtype=tf.float32)
	ns = tf.zeros((1, obs_dim), dtype=tf.float32)
	lp = tf.zeros((1, 1), dtype=tf.float32)
	_ = discriminator.logits(s, a, ns, lp)


def save_run_artifacts(output_dir, policy, discriminator, args, obs_dim, act_dim, iteration):
	os.makedirs(output_dir, exist_ok=True)

	policy_path = os.path.join(output_dir, 'policy_final.pickle')
	reward_ckpt = os.path.join(output_dir, 'reward_model_final.ckpt')
	meta_path = os.path.join(output_dir, 'run_metadata.json')

	policy_info = {
		'iteration': int(iteration),
		'algo': 'AIRL_PPO',
		'env': 'Safexp-%s%s-v0' % (args.robot_name, args.task_name),
	}
	policy.save(policy_path, policy_info)
	discriminator.save_weights(reward_ckpt)

	metadata = {
		'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
		'obs_dim': int(obs_dim),
		'act_dim': int(act_dim),
		'gamma': float(args.gamma),
		'reward_checkpoint': reward_ckpt,
		'policy_checkpoint': policy_path,
		'expert_path': args.expert_path,
		'init_policy_path': args.init_policy_path,
	}
	with open(meta_path, 'w') as f:
		json.dump(metadata, f, indent=2)

	return policy_path, reward_ckpt, meta_path


def infer_iteration_from_checkpoint_dir(checkpoint_dir):
	base = os.path.basename(os.path.normpath(checkpoint_dir))
	if base.startswith('iter_'):
		suffix = base.split('iter_', 1)[1]
		if suffix.isdigit():
			return int(suffix)

	meta_path = os.path.join(checkpoint_dir, 'run_metadata.json')
	if os.path.exists(meta_path):
		with open(meta_path, 'r') as f:
			meta = json.load(f)
		if 'iteration' in meta:
			return int(meta['iteration'])

	raise ValueError('Could not infer iteration from checkpoint dir: %s' % checkpoint_dir)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_path', type=str, required=True)
	parser.add_argument('--init_policy_path', type=str, default='')
	parser.add_argument('--robot_name', type=str, default='Point')
	parser.add_argument('--task_name', type=str, default='Goal1')
	parser.add_argument('--seed', type=int, default=0)

	parser.add_argument('--num_iterations', type=int, default=300)
	parser.add_argument('--steps_per_iter', type=int, default=2048)
	parser.add_argument('--max_ep_len', type=int, default=1000)

	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--lam', type=float, default=0.95)

	parser.add_argument('--ppo_clip_ratio', type=float, default=0.2)
	parser.add_argument('--ppo_target_kl', type=float, default=0.015)
	parser.add_argument('--ppo_train_pi_iters', type=int, default=20)
	parser.add_argument('--ppo_train_v_iters', type=int, default=20)
	parser.add_argument('--ppo_batch_size', type=int, default=256)
	parser.add_argument('--entropy_coef', type=float, default=0.0)

	parser.add_argument('--disc_updates_per_iter', type=int, default=5)
	parser.add_argument('--disc_batch_size', type=int, default=256)
	parser.add_argument('--disc_lr', type=float, default=3e-4)

	parser.add_argument('--expert_max_steps', type=int, default=0)
	parser.add_argument('--save_interval', type=int, default=25)
	parser.add_argument('--log_interval', type=int, default=10)
	parser.add_argument('--output_dir', type=str, default='')
	parser.add_argument('--resume_checkpoint_dir', type=str, default='')
	parser.add_argument('--resume_iteration', type=int, default=-1)
	return parser.parse_args()


def main():
	args = parse_args()

	np.random.seed(args.seed)
	tf.random.set_seed(args.seed)

	env_name = 'Safexp-%s%s-v0' % (args.robot_name, args.task_name)
	env = gym.make(env_name)
	print('Environment: %s' % env_name)

	expert_max_steps = args.expert_max_steps if args.expert_max_steps > 0 else None
	expert_data = load_expert_data(args.expert_path, max_steps=expert_max_steps)
	print('Loaded expert transitions: %d' % expert_data['states'].shape[0])

	obs_dim = int(expert_data['states'].shape[-1])
	act_dim = int(expert_data['actions'].shape[-1])

	policy = make_policy(obs_dim, act_dim, init_policy_path=args.init_policy_path or None)
	discriminator = AIRLDiscriminator(obs_dim, act_dim, gamma=args.gamma)
	ensure_discriminator_built(discriminator, obs_dim, act_dim)
	disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.disc_lr)

	start_iteration = 1
	if args.resume_checkpoint_dir:
		resume_dir = os.path.abspath(args.resume_checkpoint_dir)
		resume_policy_path = os.path.join(resume_dir, 'policy_final.pickle')
		resume_reward_ckpt = os.path.join(resume_dir, 'reward_model_final.ckpt')

		if not os.path.exists(resume_policy_path):
			raise ValueError('Policy checkpoint not found: %s' % resume_policy_path)
		if not os.path.exists(resume_reward_ckpt + '.index'):
			raise ValueError('Reward checkpoint not found: %s' % resume_reward_ckpt)

		print('Resuming policy from: %s' % resume_policy_path)
		print('Resuming discriminator from: %s' % resume_reward_ckpt)
		policy.load(resume_policy_path)
		discriminator.load_weights(resume_reward_ckpt)

		if args.resume_iteration >= 0:
			resume_iteration = int(args.resume_iteration)
		else:
			resume_iteration = infer_iteration_from_checkpoint_dir(resume_dir)

		start_iteration = resume_iteration + 1
		print('Resume iteration: %d (training will start at %d)' % (resume_iteration, start_iteration))

	if args.output_dir:
		output_dir = args.output_dir
	else:
		ts = time.strftime('%Y%m%d_%H%M%S')
		output_dir = os.path.join('./airl_results', 'airl_training_%s_%s%s_s%d' % (ts, args.robot_name, args.task_name, args.seed))
	os.makedirs(output_dir, exist_ok=True)

	if start_iteration > args.num_iterations:
		raise ValueError('start_iteration (%d) is greater than num_iterations (%d).' % (start_iteration, args.num_iterations))

	csv_path = os.path.join(output_dir, 'metrics.csv')
	append_metrics = bool(args.resume_checkpoint_dir and os.path.exists(csv_path))
	csv_mode = 'a' if append_metrics else 'w'
	csv_file = open(csv_path, csv_mode, newline='')
	writer = csv.writer(csv_file)
	if csv_mode == 'w':
		writer.writerow([
			'iteration', 'episodes', 'mean_ep_reward', 'mean_ep_cost', 'mean_ep_length',
			'pi_loss', 'v_loss', 'kl', 'entropy',
			'disc_loss', 'expert_acc', 'policy_acc', 'expert_prob', 'policy_prob'
		])
	csv_file.flush()

	print('Output directory: %s' % output_dir)
	start_time = time.time()

	for iteration in range(start_iteration, args.num_iterations + 1):
		buffer = RolloutBuffer(gamma=args.gamma, lam=args.lam)
		rollout_stats = collect_rollout(
			env=env,
			policy=policy,
			discriminator=discriminator,
			buffer=buffer,
			steps_per_iter=args.steps_per_iter,
			max_ep_len=args.max_ep_len,
		)
		rollout_data = buffer.get()

		ppo_stats = ppo_update(
			policy=policy,
			data=rollout_data,
			clip_ratio=args.ppo_clip_ratio,
			target_kl=args.ppo_target_kl,
			train_pi_iters=args.ppo_train_pi_iters,
			train_v_iters=args.ppo_train_v_iters,
			batch_size=args.ppo_batch_size,
			entropy_coef=args.entropy_coef,
		)

		disc_logs = []
		for _ in range(args.disc_updates_per_iter):
			dlog = discriminator_update(
				discriminator=discriminator,
				optimizer=disc_optimizer,
				expert_data=expert_data,
				policy_data=rollout_data,
				policy=policy,
				batch_size=args.disc_batch_size,
			)
			disc_logs.append(dlog)

		disc_stats = {}
		keys = disc_logs[0].keys() if disc_logs else []
		for k in keys:
			disc_stats[k] = float(np.mean([x[k] for x in disc_logs]))

		writer.writerow([
			iteration,
			rollout_stats['episodes'],
			rollout_stats['mean_ep_reward'],
			rollout_stats['mean_ep_cost'],
			rollout_stats['mean_ep_length'],
			ppo_stats['pi_loss'],
			ppo_stats['v_loss'],
			ppo_stats['kl'],
			ppo_stats['entropy'],
			disc_stats.get('disc_loss', 0.0),
			disc_stats.get('expert_acc', 0.0),
			disc_stats.get('policy_acc', 0.0),
			disc_stats.get('expert_prob', 0.0),
			disc_stats.get('policy_prob', 0.0),
		])
		csv_file.flush()

		if iteration % args.log_interval == 0 or iteration == 1:
			elapsed = time.time() - start_time
			print(
				'Iter %d | Episodes %d | Reward %.3f | Cost %.3f | PiLoss %.4f | VLoss %.4f | KL %.4f | DLoss %.4f | Time %.1fs'
				% (
					iteration,
					rollout_stats['episodes'],
					rollout_stats['mean_ep_reward'],
					rollout_stats['mean_ep_cost'],
					ppo_stats['pi_loss'],
					ppo_stats['v_loss'],
					ppo_stats['kl'],
					disc_stats.get('disc_loss', 0.0),
					elapsed,
				)
			)

		if iteration % args.save_interval == 0:
			iter_dir = os.path.join(output_dir, 'checkpoints', 'iter_%06d' % iteration)
			save_run_artifacts(iter_dir, policy, discriminator, args, obs_dim, act_dim, iteration)

	policy_path, reward_ckpt, meta_path = save_run_artifacts(output_dir, policy, discriminator, args, obs_dim, act_dim, args.num_iterations)

	csv_file.close()
	env.close()

	print('Training complete.')
	print('Policy saved to: %s' % policy_path)
	print('Reward model saved to: %s' % reward_ckpt)
	print('Run metadata saved to: %s' % meta_path)
	print('Metrics CSV saved to: %s' % csv_path)


if __name__ == '__main__':
	main()
