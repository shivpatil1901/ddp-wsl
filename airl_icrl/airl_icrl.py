"""
Active AIRL + PPO-Lagrangian training pipeline.

This script wires together:
1) Safety environment instantiation (SafetyPointGoal variants).
2) Expert policy wrapper for exact action log-probability queries.
3) Regression-style AIRL reward model updates.
4) PPO-Lagrangian generator updates with dual (reward/cost) GAE.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def safe_reset(env):
	out = env.reset()
	if isinstance(out, tuple):
		return out[0]
	return out


def safe_step(env, action):
	out = env.step(action)
	if len(out) == 5:
		next_state, reward, terminated, truncated, info = out
		done = bool(terminated or truncated)
		return next_state, reward, done, info
	return out


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Linear(input_dim, hidden_dim),
		nn.ReLU(),
		nn.Linear(hidden_dim, hidden_dim),
		nn.ReLU(),
		nn.Linear(hidden_dim, output_dim),
	)


class ActorNet(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
		super().__init__()
		self.mean_net = build_mlp(state_dim, hidden_size, action_dim)
		self.log_std = nn.Parameter(torch.zeros(action_dim))

	def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		mean = self.mean_net(states)
		log_std = self.log_std.expand_as(mean)
		return mean, log_std


class CriticNet(nn.Module):
	def __init__(self, state_dim: int, hidden_size: int = 256):
		super().__init__()
		self.value_net = build_mlp(state_dim, hidden_size, 1)

	def forward(self, states: torch.Tensor) -> torch.Tensor:
		return self.value_net(states).squeeze(-1)


class RewardNet(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
		super().__init__()
		self.net = build_mlp(state_dim * 2 + action_dim, hidden_size, 1)

	def forward(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
		x = torch.cat([states, actions, next_states], dim=-1)
		return self.net(x)


class ExpertWrapper:
	"""Loads expert actor weights and returns log pi_E(a|s)."""

	def __init__(
		self,
		pre_trained_actor_network: Optional[nn.Module],
		device: str = "cpu",
		tf_actor=None,
		tf_module=None,
	):
		self.device = torch.device(device)
		self.actor = None
		self.tf_actor = tf_actor
		self.tf = tf_module

		if pre_trained_actor_network is not None:
			self.actor = pre_trained_actor_network.to(self.device)
			self.actor.eval()
			for p in self.actor.parameters():
				p.requires_grad_(False)

	@classmethod
	def _resolve_checkpoint_path(cls, checkpoint_path: str) -> str:
		raw = os.path.expanduser(os.path.expandvars(checkpoint_path))
		normalized = raw.replace('\\', os.sep).replace('/', os.sep)

		candidates = []
		if os.path.isabs(normalized):
			candidates.append(normalized)
		else:
			candidates.append(os.path.abspath(normalized))
			repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
			candidates.append(os.path.abspath(os.path.join(repo_root, normalized)))

		for path in candidates:
			if os.path.exists(path):
				return path

		raise FileNotFoundError(
			'Expert checkpoint not found: %s. Tried: %s' % (checkpoint_path, ', '.join(candidates))
		)

	@classmethod
	def from_checkpoint(
		cls,
		checkpoint_path: str,
		state_dim: int,
		action_dim: int,
		hidden_size: int = 256,
		device: str = "cpu",
	) -> "ExpertWrapper":
		resolved_checkpoint_path = cls._resolve_checkpoint_path(checkpoint_path)

		# Try native PyTorch checkpoints first.
		actor = ActorNet(state_dim, action_dim, hidden_size=hidden_size)
		try:
			checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
		except RuntimeError as err:
			if 'Invalid magic number' not in str(err):
				raise
			return cls._from_safedice_checkpoint(
				resolved_checkpoint_path,
				state_dim=state_dim,
				action_dim=action_dim,
				hidden_size=hidden_size,
				device=device,
			)

		state_dict = checkpoint
		if isinstance(checkpoint, dict):
			candidate_keys = [
				"actor_state_dict",
				"actor",
				"state_dict",
				"model_state_dict",
			]
			for key in candidate_keys:
				if key in checkpoint and isinstance(checkpoint[key], dict):
					state_dict = checkpoint[key]
					break

		actor.load_state_dict(state_dict)
		return cls(actor, device=device)

	@classmethod
	def _from_safedice_checkpoint(
		cls,
		checkpoint_path: str,
		state_dim: int,
		action_dim: int,
		hidden_size: int,
		device: str,
	) -> "ExpertWrapper":
		repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		safedice_root = os.path.join(repo_root, 'SafeDICE')
		if safedice_root not in sys.path:
			sys.path.append(safedice_root)

		import tensorflow as tf
		from algorithms.safedice import SafeDICE as AntiDICE

		config = {
			'hidden_size': int(hidden_size),
			'critic_lr': 3e-4,
			'actor_lr': 3e-4,
			'grad_reg_coeffs': [10.0, 10.0],
			'gamma': 0.99,
			'alpha': 0,
			'use_last_layer_bias_cost': True,
			'use_last_layer_bias_critic': True,
			'kernel_initializer': 'glorot_uniform',
		}
		expert_policy = AntiDICE(
			state_dim,
			action_dim,
			mixture_actor=False,
			is_discrete_action=False,
			config=config,
		)
		expert_policy.load(checkpoint_path)
		return cls(pre_trained_actor_network=None, device=device, tf_actor=expert_policy.actor, tf_module=tf)

	def evaluate_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
		"""Returns expert log-probability log pi_E(a|s)."""
		if self.tf_actor is not None:
			states_np = states.detach().cpu().numpy().astype(np.float32)
			actions_np = actions.detach().cpu().numpy().astype(np.float32)
			states_tf = self.tf.convert_to_tensor(states_np, dtype=self.tf.float32)
			actions_tf = self.tf.convert_to_tensor(actions_np, dtype=self.tf.float32)
			log_prob_tf = self.tf_actor.get_log_prob(states_tf, actions_tf)
			log_prob_np = np.asarray(log_prob_tf.numpy(), dtype=np.float32)
			return torch.as_tensor(log_prob_np, dtype=torch.float32, device=states.device)

		states = states.to(self.device)
		actions = actions.to(self.device)
		with torch.no_grad():
			mean, log_std = self.actor(states)
			std = log_std.exp()
			dist = Normal(mean, std)
			log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
		return log_prob


class RegressionAIRL:
	"""Structurally disentangled AIRL module to isolate r(s,a) from V(s)."""

	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		lr: float = 3e-4,
		hidden_size: int = 256,
		gamma: float = 0.99,
		device: str = "cpu",
	):
		self.device = torch.device(device)
		self.gamma = float(gamma)

		self.r_net = build_mlp(state_dim + action_dim, hidden_size, 1).to(self.device)
		self.v_net = build_mlp(state_dim, hidden_size, 1).to(self.device)

		self.optimizer = optim.Adam(
			list(self.r_net.parameters()) + list(self.v_net.parameters()),
			lr=lr,
		)

	def update(
		self,
		states: torch.Tensor,
		actions: torch.Tensor,
		next_states: torch.Tensor,
		costs: torch.Tensor,
		expert_log_probs: torch.Tensor,
		current_lambda: float,
	) -> Dict[str, float]:
		states = states.to(self.device)
		actions = actions.to(self.device)
		next_states = next_states.to(self.device)
		costs = costs.to(self.device)
		expert_log_probs = expert_log_probs.to(self.device)
		expert_log_probs = torch.clamp(expert_log_probs, min=-20.0, max=50.0)

		state_action = torch.cat([states, actions], dim=-1)
		r_pred = self.r_net(state_action)
		v_s = self.v_net(states)
		v_next = self.v_net(next_states)

		# A_pred = r(s,a) - lambda*c + gamma*V(s') - V(s)
		adv_pred = r_pred - (float(current_lambda) * costs) + (self.gamma * v_next) - v_s
		loss = F.mse_loss(adv_pred, expert_log_probs)

		# Compute component-aligned diagnostic losses (monitoring only).
		# r_net is evaluated against the residual target after removing current v-term.
		# v_net is evaluated against the residual target after removing current r-term.
		with torch.no_grad():
			v_diff = (self.gamma * v_next) - v_s
			r_target = expert_log_probs + (float(current_lambda) * costs) - v_diff
			v_target = expert_log_probs - r_pred + (float(current_lambda) * costs)
			r_loss = F.mse_loss(r_pred, r_target)
			v_loss = F.mse_loss(v_diff, v_target)

		self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.r_net.parameters(), max_norm=0.5)
		nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=0.5)
		self.optimizer.step()

		mean_reward = float(r_pred.mean().detach().cpu().item())
		return {
			'total_loss': float(loss.detach().cpu().item()),
			'r_loss': float(r_loss.detach().cpu().item()),
			'v_loss': float(v_loss.detach().cpu().item()),
			'mean_reward': mean_reward,
		}

	def get_reward(
		self,
		states: torch.Tensor,
		actions: torch.Tensor,
		next_states: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		states = states.to(self.device)
		actions = actions.to(self.device)
		state_action = torch.cat([states, actions], dim=-1)
		return self.r_net(state_action)


class PPOLagGenerator:
	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		cost_limit: float,
		lr: float = 3e-4,
		gamma: float = 0.99,
		lam: float = 0.95,
		hidden_size: int = 256,
		clip_ratio: float = 0.2,
		target_kl: float = 0.015,
		train_pi_iters: int = 20,
		train_v_iters: int = 20,
		batch_size: int = 256,
		entropy_coef: float = 0.0,
		max_grad_norm: float = 0.5,
		lambda_lr: Optional[float] = None,
		device: str = "cpu",
	):
		self.device = torch.device(device)

		self.actor = ActorNet(state_dim, action_dim, hidden_size=hidden_size).to(self.device)
		self.reward_critic = CriticNet(state_dim, hidden_size=hidden_size).to(self.device)
		self.cost_critic = CriticNet(state_dim, hidden_size=hidden_size).to(self.device)

		self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
		self.reward_critic_opt = optim.Adam(self.reward_critic.parameters(), lr=lr)
		self.cost_critic_opt = optim.Adam(self.cost_critic.parameters(), lr=lr)

		self.log_lambda = torch.zeros(1, requires_grad=True, device=self.device)
		# Use 10x higher LR for lambda: dual ascent needs aggressive multiplier updates
		effective_lambda_lr = lambda_lr if lambda_lr is not None else (lr * 10.0)
		self.lambda_opt = optim.Adam([self.log_lambda], lr=effective_lambda_lr)

		self.cost_limit = float(cost_limit)
		self.gamma = float(gamma)
		self.lam = float(lam)
		self.clip_ratio = float(clip_ratio)
		self.target_kl = float(target_kl)
		self.train_pi_iters = int(train_pi_iters)
		self.train_v_iters = int(train_v_iters)
		self.batch_size = int(batch_size)
		self.entropy_coef = float(entropy_coef)
		self.max_grad_norm = float(max_grad_norm)

	def _dist(self, states: torch.Tensor) -> Normal:
		mean, log_std = self.actor(states)
		return Normal(mean, log_std.exp())

	def evaluate_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
		dist = self._dist(states)
		return dist.log_prob(actions).sum(dim=-1)

	def get_action(self, state: np.ndarray) -> np.ndarray:
		with torch.no_grad():
			state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
			dist = self._dist(state_t)
			action = dist.sample()
		return action.squeeze(0).cpu().numpy()

	def get_current_lambda(self) -> float:
		return float(F.softplus(self.log_lambda).detach().cpu().item())

	def _compute_gae(
		self,
		rewards: torch.Tensor,
		values: torch.Tensor,
		next_values: torch.Tensor,
		dones: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		advantages = torch.zeros_like(rewards)
		gae = torch.zeros(1, dtype=rewards.dtype, device=rewards.device)
		for t in reversed(range(rewards.shape[0])):
			not_done = 1.0 - dones[t]
			delta = rewards[t] + self.gamma * next_values[t] * not_done - values[t]
			gae = delta + self.gamma * self.lam * not_done * gae
			advantages[t] = gae
		returns = advantages + values
		return advantages, returns

	def update(
		self,
		states: torch.Tensor,
		actions: torch.Tensor,
		learned_rewards: torch.Tensor,
		costs: torch.Tensor,
		next_states: Optional[torch.Tensor] = None,
		dones: Optional[torch.Tensor] = None,
		) -> Tuple[float, float]:
		states = states.to(self.device)
		actions = actions.to(self.device)
		learned_rewards = learned_rewards.to(self.device).view(-1)
		costs = costs.to(self.device).view(-1)

		if next_states is None:
			next_states = states
		if dones is None:
			dones = torch.zeros_like(learned_rewards)

		next_states = next_states.to(self.device)
		dones = dones.to(self.device).view(-1)

		# Dual ascent on lambda to enforce the expected-cost constraint.
		lambda_val = F.softplus(self.log_lambda)
		episode_costs = []
		running_episode_cost = torch.zeros(1, dtype=costs.dtype, device=costs.device)
		steps_in_episode = 0
		for t in range(costs.shape[0]):
			running_episode_cost = running_episode_cost + costs[t]
			steps_in_episode += 1
			if dones[t] > 0.5:
				episode_costs.append(running_episode_cost)
				running_episode_cost = torch.zeros(1, dtype=costs.dtype, device=costs.device)
				steps_in_episode = 0

		if steps_in_episode > 0:
			episode_costs.append(running_episode_cost)

		if episode_costs:
			mean_episode_cost = torch.stack(episode_costs).mean()
		else:
			mean_episode_cost = costs.sum()

		lambda_loss = -(lambda_val * (mean_episode_cost - self.cost_limit).detach())

		self.lambda_opt.zero_grad()
		lambda_loss.backward()
		if self.log_lambda.grad is not None:
			# Debug: show constraint violation and gradient magnitude
			constraint_violation = float((mean_episode_cost - self.cost_limit).detach().cpu().item())
			grad_magnitude = float(self.log_lambda.grad.abs().detach().cpu().item())
			if constraint_violation > 1.0:  # Only print when violation is significant
				print(f"  [Lambda Debug] Constraint violation: {constraint_violation:.2f}, Gradient: {grad_magnitude:.6f}, Current lambda: {float(lambda_val.detach().cpu().item()):.6f}")
		self.lambda_opt.step()

		with torch.no_grad():
			old_log_probs = self.evaluate_log_prob(states, actions)

			reward_vals = self.reward_critic(states)
			reward_next_vals = self.reward_critic(next_states)
			cost_vals = self.cost_critic(states)
			cost_next_vals = self.cost_critic(next_states)

			adv_r, ret_r = self._compute_gae(learned_rewards, reward_vals, reward_next_vals, dones)
			adv_c, ret_c = self._compute_gae(costs, cost_vals, cost_next_vals, dones)

			current_lambda = F.softplus(self.log_lambda).detach()
			adv_lagrangian = adv_r - current_lambda * adv_c
			adv_lagrangian = (adv_lagrangian - adv_lagrangian.mean()) / (adv_lagrangian.std() + 1e-8)

		n = states.shape[0]
		batch_size = min(max(1, self.batch_size), n)

		actor_losses = []
		approx_kls = []
		entropies = []
		reward_critic_losses = []
		cost_critic_losses = []

		# Standard PPO clipped surrogate optimization.
		for _ in range(self.train_pi_iters):
			perm = torch.randperm(n, device=self.device)
			stop_early = False

			for start in range(0, n, batch_size):
				idx = perm[start:start + batch_size]
				b_states = states[idx]
				b_actions = actions[idx]
				b_old_log_probs = old_log_probs[idx]
				b_adv = adv_lagrangian[idx]

				dist = self._dist(b_states)
				new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
				entropy = dist.entropy().sum(dim=-1).mean()

				ratio = torch.exp(new_log_probs - b_old_log_probs)
				surr_1 = ratio * b_adv
				surr_2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
				actor_loss = -torch.min(surr_1, surr_2).mean() - self.entropy_coef * entropy

				self.actor_opt.zero_grad()
				actor_loss.backward()
				nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				self.actor_opt.step()

				approx_kl = (b_old_log_probs - new_log_probs).mean()

				actor_losses.append(float(actor_loss.detach().cpu().item()))
				approx_kls.append(float(approx_kl.detach().cpu().item()))
				entropies.append(float(entropy.detach().cpu().item()))

				if float(approx_kl.detach().cpu().item()) > 1.5 * self.target_kl:
					stop_early = True
					break

			if stop_early:
				break

		for _ in range(self.train_v_iters):
			perm = torch.randperm(n, device=self.device)

			for start in range(0, n, batch_size):
				idx = perm[start:start + batch_size]
				b_states = states[idx]
				b_ret_r = ret_r[idx]
				b_ret_c = ret_c[idx]

				pred_r = self.reward_critic(b_states)
				loss_r = F.mse_loss(pred_r, b_ret_r)
				self.reward_critic_opt.zero_grad()
				loss_r.backward()
				nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.max_grad_norm)
				self.reward_critic_opt.step()
				reward_critic_losses.append(float(loss_r.detach().cpu().item()))

				pred_c = self.cost_critic(b_states)
				loss_c = F.mse_loss(pred_c, b_ret_c)
				self.cost_critic_opt.zero_grad()
				loss_c.backward()
				nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
				self.cost_critic_opt.step()
				cost_critic_losses.append(float(loss_c.detach().cpu().item()))

		return {
			"actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
			"reward_critic_loss": float(np.mean(reward_critic_losses)) if reward_critic_losses else 0.0,
			"cost_critic_loss": float(np.mean(cost_critic_losses)) if cost_critic_losses else 0.0,
			"entropy": float(np.mean(entropies)) if entropies else 0.0,
			"approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
			"lambda": self.get_current_lambda(),
			"lambda_loss": float(lambda_loss.detach().cpu().item()),
			"mean_cost": float(mean_episode_cost.detach().cpu().item()),
			"mean_step_cost": float(costs.mean().detach().cpu().item()),
		}


def make_run_output_dir(base_output_dir: str, run_name: str, robot_name: str, task_name: str, seed: int) -> str:
	ts = time.strftime('%Y%m%d_%H%M%S')
	if run_name:
		folder_name = run_name
	else:
		folder_name = 'active_airl_%s_%s%s_s%d' % (ts, robot_name, task_name, seed)
	out_dir = os.path.join(base_output_dir, folder_name)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


def save_training_checkpoint(
	checkpoint_path: str,
	epoch: int,
	generator: PPOLagGenerator,
	airl_module: RegressionAIRL,
	extra_metadata: Optional[Dict] = None,
):
	payload = {
		'epoch': int(epoch),
		'generator': {
			'actor_state_dict': generator.actor.state_dict(),
			'reward_critic_state_dict': generator.reward_critic.state_dict(),
			'cost_critic_state_dict': generator.cost_critic.state_dict(),
			'log_lambda': generator.log_lambda.detach().cpu(),
		},
		'airl_module': {
			# Keep reward_net_state_dict as a compatibility alias for older tooling.
			'reward_net_state_dict': airl_module.r_net.state_dict(),
			'r_net_state_dict': airl_module.r_net.state_dict(),
			'v_net_state_dict': airl_module.v_net.state_dict(),
			'architecture': 'disentangled_r_v',
		},
		'optimizers': {
			'actor_opt_state_dict': generator.actor_opt.state_dict(),
			'reward_critic_opt_state_dict': generator.reward_critic_opt.state_dict(),
			'cost_critic_opt_state_dict': generator.cost_critic_opt.state_dict(),
			'lambda_opt_state_dict': generator.lambda_opt.state_dict(),
			'airl_opt_state_dict': airl_module.optimizer.state_dict(),
		},
		'metadata': extra_metadata or {},
	}
	torch.save(payload, checkpoint_path)


def train_active_irl(
	env,
	expert_policy: ExpertWrapper,
	generator: PPOLagGenerator,
	airl_module: RegressionAIRL,
	num_epochs: int = 500,
	steps_per_epoch: int = 4000,
	output_dir: Optional[str] = None,
	save_interval: int = 0,
	run_metadata: Optional[Dict] = None,
):
	device = generator.device

	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
		print('Checkpoint directory: %s' % os.path.abspath(output_dir))

	for epoch in range(num_epochs):
		states, actions, next_states, costs, dones = [], [], [], [], []
		state = safe_reset(env)

		for _ in range(steps_per_epoch):
			action = generator.get_action(state)

			if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
				action = np.clip(action, env.action_space.low, env.action_space.high)

			next_state, _env_reward, done, info = safe_step(env, action)
			cost = float(info.get("cost", 0.0))

			states.append(state)
			actions.append(action)
			next_states.append(next_state)
			costs.append(cost)
			dones.append(float(done))

			state = safe_reset(env) if done else next_state

		states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
		actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
		next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device)
		costs_t = torch.as_tensor(np.asarray(costs), dtype=torch.float32, device=device).unsqueeze(1)
		dones_t = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device)

		with torch.no_grad():
			expert_log_probs = expert_policy.evaluate_log_prob(states_t, actions_t)

		current_lambda = generator.get_current_lambda()
		irl_result = airl_module.update(
			states_t,
			actions_t,
			next_states_t,
			costs_t,
			expert_log_probs,
			current_lambda,
		)
		irl_loss = irl_result['total_loss']
		r_loss = irl_result['r_loss']
		v_loss = irl_result['v_loss']
		mean_learned_reward = irl_result['mean_reward']

		with torch.no_grad():
			learned_rewards = airl_module.get_reward(states_t, actions_t, next_states_t).squeeze(-1)

		generator_loss_info = generator.update(
			states=states_t,
			actions=actions_t,
			learned_rewards=learned_rewards,
			costs=costs_t.squeeze(-1),
			next_states=next_states_t,
			dones=dones_t,
		)

		if epoch % 10 == 0:
			print(
				"Epoch: %d | IRL Loss: %.4f (r_net: %.4f, v_net: %.4f) | Mean Learned Reward: %.4f | "
				"Lambda: %.4f | Actor Loss: %.4f | Mean Episode Cost: %.4f | Mean Step Cost: %.4f"
				% (
					epoch,
					irl_loss,
					r_loss,
					v_loss,
					mean_learned_reward,
					generator_loss_info.get("lambda", current_lambda),
					generator_loss_info.get("actor_loss", 0.0),
					generator_loss_info.get("mean_cost", 0.0),
					generator_loss_info.get("mean_step_cost", 0.0),
				)
			)

		if output_dir and save_interval > 0 and (epoch + 1) % save_interval == 0:
			ckpt_path = os.path.join(output_dir, 'checkpoint_epoch_%06d.pt' % (epoch + 1))
			save_training_checkpoint(
				checkpoint_path=ckpt_path,
				epoch=epoch + 1,
				generator=generator,
				airl_module=airl_module,
				extra_metadata={
					'run_metadata': run_metadata or {},
					'metrics': {
						'irl_loss': float(irl_loss),
						'r_loss': float(r_loss),
						'v_loss': float(v_loss),
						'mean_learned_reward': float(mean_learned_reward),
						'lambda': float(generator_loss_info.get('lambda', current_lambda)),
						'actor_loss': float(generator_loss_info.get('actor_loss', 0.0)),
						'mean_episode_cost': float(generator_loss_info.get('mean_cost', 0.0)),
						'mean_step_cost': float(generator_loss_info.get('mean_step_cost', 0.0)),
					},
				},
			)
			print('Saved checkpoint: %s' % ckpt_path)

	if output_dir:
		final_ckpt_path = os.path.join(output_dir, 'checkpoint_final.pt')
		save_training_checkpoint(
			checkpoint_path=final_ckpt_path,
			epoch=num_epochs,
			generator=generator,
			airl_module=airl_module,
			extra_metadata={
				'run_metadata': run_metadata or {},
				'is_final': True,
			},
		)
		print('Saved final checkpoint: %s' % final_ckpt_path)


def make_environment(env_backend: str, env_id: str, robot_name: str, task_name: str):
	if env_backend in ("auto", "gymnasium"):
		try:
			import safety_gymnasium

			resolved_env_id = env_id or "SafetyPointGoal1-v0"
			env = safety_gymnasium.make(resolved_env_id)
			return env, resolved_env_id
		except Exception:
			if env_backend == "gymnasium":
				raise

	if env_backend in ("auto", "gym"):
		import gym
		import safety_gym  # noqa: F401

		resolved_env_id = env_id or ("Safexp-%s%s-v0" % (robot_name, task_name))
		env = gym.make(resolved_env_id)
		return env, resolved_env_id

	raise ValueError("Unsupported env backend: %s" % env_backend)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_backend", type=str, default="auto", choices=["auto", "gymnasium", "gym"])
	parser.add_argument("--env_id", type=str, default="")
	parser.add_argument("--robot_name", type=str, default="Point")
	parser.add_argument("--task_name", type=str, default="Goal1")

	parser.add_argument("--expert_checkpoint", type=str, required=True)
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--num_epochs", type=int, default=500)
	parser.add_argument("--steps_per_epoch", type=int, default=4000)
	parser.add_argument("--output_dir", type=str, default="./airl_icrl_checkpoints")
	parser.add_argument("--run_name", type=str, default="")
	parser.add_argument("--save_interval", type=int, default=50)

	parser.add_argument("--cost_limit", type=float, default=25.0)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--lam", type=float, default=0.95)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--lambda_lr", type=float, default=None, help="Learning rate for lambda (Lagrange multiplier). Defaults to 10x main lr for aggressive dual ascent.")
	parser.add_argument("--airl_lr", type=float, default=3e-4)

	parser.add_argument("--ppo_clip_ratio", type=float, default=0.2)
	parser.add_argument("--ppo_target_kl", type=float, default=0.015)
	parser.add_argument("--ppo_train_pi_iters", type=int, default=20)
	parser.add_argument("--ppo_train_v_iters", type=int, default=20)
	parser.add_argument("--ppo_batch_size", type=int, default=256)
	parser.add_argument("--entropy_coef", type=float, default=0.0)
	parser.add_argument("--max_grad_norm", type=float, default=0.5)

	return parser.parse_args()


def main():
	args = parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	env, resolved_env_id = make_environment(
		env_backend=args.env_backend,
		env_id=args.env_id,
		robot_name=args.robot_name,
		task_name=args.task_name,
	)
	print("Environment: %s" % resolved_env_id)
	run_output_dir = make_run_output_dir(
		base_output_dir=args.output_dir,
		run_name=args.run_name,
		robot_name=args.robot_name,
		task_name=args.task_name,
		seed=args.seed,
	)
	print('Run output directory: %s' % run_output_dir)

	state_dim = int(env.observation_space.shape[0])
	action_dim = int(env.action_space.shape[0])

	expert = ExpertWrapper.from_checkpoint(
		checkpoint_path=args.expert_checkpoint,
		state_dim=state_dim,
		action_dim=action_dim,
		hidden_size=args.hidden_size,
		device=args.device,
	)

	generator = PPOLagGenerator(
		state_dim=state_dim,
		action_dim=action_dim,
		cost_limit=args.cost_limit,
		lr=args.lr,
		gamma=args.gamma,
		lam=args.lam,
		hidden_size=args.hidden_size,
		clip_ratio=args.ppo_clip_ratio,
		target_kl=args.ppo_target_kl,
		train_pi_iters=args.ppo_train_pi_iters,
		train_v_iters=args.ppo_train_v_iters,
		batch_size=args.ppo_batch_size,
		entropy_coef=args.entropy_coef,
		max_grad_norm=args.max_grad_norm,
		lambda_lr=args.lambda_lr,
		device=args.device,
	)

	airl_module = RegressionAIRL(
		state_dim=state_dim,
		action_dim=action_dim,
		lr=args.airl_lr,
		hidden_size=args.hidden_size,
		gamma=args.gamma,
		device=args.device,
	)

	train_active_irl(
		env=env,
		expert_policy=expert,
		generator=generator,
		airl_module=airl_module,
		num_epochs=args.num_epochs,
		steps_per_epoch=args.steps_per_epoch,
		output_dir=run_output_dir,
		save_interval=args.save_interval,
		run_metadata={
			'args': vars(args),
			'env_id': resolved_env_id,
			'state_dim': state_dim,
			'action_dim': action_dim,
		},
	)

	env.close()


if __name__ == "__main__":
	main()
