"""
Build a reward-balanced trajectory dataset with two pools:
- 900 low-reward trajectories (< 15 cumulative reward) generated from a noisy policy
- 900 high-reward trajectories (> 15 cumulative reward) sampled from a SafeDICE dataset

Default policy:
data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0/simple_save332

Default SafeDICE dataset:
SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
	import pickle5 as pickle
except ImportError:
	import pickle


DEFAULT_POLICY_PATH = "data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0/simple_save332"
DEFAULT_SAFEDICE_PATH = "SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
DEFAULT_OUTPUT_PATH = "rudder/reward_balanced_1800.pkl"


def _repo_root() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)
	if os.path.isabs(normalized):
		return normalized
	return os.path.abspath(os.path.join(_repo_root(), normalized))


def _pick(mapping: Dict[str, Any], candidates: Sequence[str]) -> Optional[Any]:
	for key in candidates:
		if key in mapping:
			return mapping[key]
	return None


def _as_np32(arr: Any) -> np.ndarray:
	return np.asarray(arr, dtype=np.float32)


def _as_1d(arr: Any) -> np.ndarray:
	return np.asarray(arr).reshape(-1)


def _safe_reset(env):
	out = env.reset()
	if isinstance(out, tuple):
		return out[0]
	return out


def _safe_step(env, action):
	out = env.step(action)
	if len(out) == 5:
		next_obs, reward, terminated, truncated, info = out
		done = bool(terminated or truncated)
		return next_obs, reward, done, info
	return out


def _trajectory_return(traj: Dict[str, Any]) -> float:
	rewards = _pick(traj, ("rewards", "reward", "rews", "r", "env_rewards"))
	if rewards is None:
		raise KeyError("Trajectory missing rewards/reward/rews/r field")
	return float(np.sum(_as_1d(rewards).astype(np.float32)))


def _extract_payload(loaded: Any) -> Any:
	payload = loaded
	if isinstance(payload, dict):
		ts = payload.get("training_state")
		if isinstance(ts, dict):
			for key in ("dataset", "data", "replay_buffer", "buffer", "expert_data", "trajectories"):
				if key in ts:
					payload = ts[key]
					break

		if payload is loaded:
			for key in ("dataset", "data", "expert_data", "trajectories"):
				if key in payload:
					payload = payload[key]
					break

	return payload


def _split_flat_dataset_into_trajectories(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
	states = _pick(payload, ("states", "observations", "obs", "state", "s"))
	actions = _pick(payload, ("actions", "acts", "action", "a"))
	next_states = _pick(payload, ("next_states", "next_observations", "next_obs", "next_state", "s_next", "obs2"))
	rewards = _pick(payload, ("rewards", "reward", "rews", "r", "env_rewards"))
	costs = _pick(payload, ("costs", "cost", "c"))
	dones = _pick(payload, ("dones", "done", "terminals", "terminal", "episode_ends", "timeouts"))

	if rewards is None:
		raise KeyError("No reward field found in flat dataset")
	if dones is None:
		raise KeyError("No done/terminal field found in flat dataset; cannot segment trajectories")

	arrays: Dict[str, np.ndarray] = {}
	if states is not None:
		arrays["states"] = _as_np32(states)
	if actions is not None:
		arrays["actions"] = _as_np32(actions)
	if next_states is not None:
		arrays["next_states"] = _as_np32(next_states)
	arrays["rewards"] = _as_np32(rewards)
	if costs is not None:
		arrays["costs"] = _as_np32(costs)

	done_arr = _as_1d(dones).astype(bool)

	lengths = [len(done_arr)] + [len(v) for v in arrays.values()]
	n = int(min(lengths))
	done_arr = done_arr[:n]
	for key in list(arrays.keys()):
		arrays[key] = arrays[key][:n]

	trajectories: List[Dict[str, np.ndarray]] = []
	start = 0
	for i, done in enumerate(done_arr):
		if done:
			if i + 1 > start:
				traj = {k: v[start:i + 1] for k, v in arrays.items()}
				traj["dones"] = done_arr[start:i + 1].astype(np.float32)
				trajectories.append(traj)
			start = i + 1

	if start < n:
		traj = {k: v[start:n] for k, v in arrays.items()}
		traj["dones"] = done_arr[start:n].astype(np.float32)
		trajectories.append(traj)

	return trajectories


def _extract_trajectories(loaded: Any) -> List[Dict[str, Any]]:
	payload = _extract_payload(loaded)

	if isinstance(payload, list):
		trajectories = [t for t in payload if isinstance(t, dict)]
		if not trajectories:
			raise ValueError("Dataset payload is a list but contains no trajectory dicts")
		return trajectories

	if isinstance(payload, dict):
		return _split_flat_dataset_into_trajectories(payload)

	raise TypeError("Unsupported dataset payload type: %s" % type(payload))


def _policy_root_and_itr(policy_path: str) -> Tuple[str, Union[str, int]]:
	path = _resolve_path(policy_path)
	if not os.path.exists(path):
		raise FileNotFoundError("Policy path not found: %s" % path)

	base = os.path.basename(path)
	if os.path.isdir(path) and base.startswith("simple_save"):
		suffix = base[len("simple_save"):]
		if suffix.isdigit():
			return os.path.dirname(path), int(suffix)
		return os.path.dirname(path), "last"

	return path, "last"


def _load_policy(policy_path: str, deterministic: bool):
	starter_root = os.path.join(_repo_root(), "3rdparty", "safety-starter-agents")
	if starter_root not in sys.path:
		sys.path.insert(0, starter_root)

	from safe_rl.utils.load_utils import load_policy  # pylint: disable=import-error

	policy_root, itr = _policy_root_and_itr(policy_path)
	env, get_action, sess = load_policy(policy_root, itr=itr, deterministic=deterministic)
	return env, get_action, sess, policy_root, itr


def _build_env_if_missing(env, env_id: str):
	if env is not None:
		return env

	import gym
	import safety_gym  # noqa: F401

	return gym.make(env_id)


def _collect_noisy_low_reward(
	env,
	get_action,
	target_count: int,
	reward_threshold: float,
	noise_std: float,
	max_rollout_episodes: int,
	max_ep_len: int,
	seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
	rng = np.random.RandomState(seed)
	kept: List[Dict[str, np.ndarray]] = []
	attempted = 0

	while attempted < max_rollout_episodes and len(kept) < target_count:
		obs = _safe_reset(env)
		ep_states = []
		ep_actions = []
		ep_next_states = []
		ep_rewards = []
		ep_costs = []
		ep_dones = []

		ep_ret = 0.0
		ep_cost = 0.0
		ep_len = 0
		done = False

		while not done and ep_len < max_ep_len:
			action = np.asarray(get_action(obs), dtype=np.float32)
			if noise_std > 0.0:
				action = action + rng.normal(loc=0.0, scale=noise_std, size=action.shape).astype(np.float32)

			if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
				action = np.clip(action, env.action_space.low, env.action_space.high)

			next_obs, reward, done, info = _safe_step(env, action)
			cost = float(info.get("cost", 0.0))

			ep_states.append(obs)
			ep_actions.append(action)
			ep_next_states.append(next_obs)
			ep_rewards.append(float(reward))
			ep_costs.append(cost)
			ep_dones.append(float(done))

			ep_ret += float(reward)
			ep_cost += cost
			ep_len += 1
			obs = next_obs

		attempted += 1
		keep = ep_ret < reward_threshold
		if keep:
			kept.append(
				{
					"states": np.asarray(ep_states, dtype=np.float32),
					"actions": np.asarray(ep_actions, dtype=np.float32),
					"next_states": np.asarray(ep_next_states, dtype=np.float32),
					"rewards": np.asarray(ep_rewards, dtype=np.float32),
					"costs": np.asarray(ep_costs, dtype=np.float32),
					"dones": np.asarray(ep_dones, dtype=np.float32),
					"source": np.asarray([0], dtype=np.int32),
				}
			)

		print(
			"Noisy rollout %d | return=%.3f cost=%.3f len=%d | keep(<%.2f)=%s | kept=%d/%d"
			% (attempted, ep_ret, ep_cost, ep_len, reward_threshold, str(keep), len(kept), target_count)
		)

	if len(kept) < target_count:
		raise RuntimeError(
			"Collected only %d low-reward trajectories (target=%d) after %d episodes. "
			"Increase --max_rollout_episodes, increase --noise_std, or relax --reward_threshold."
			% (len(kept), target_count, attempted)
		)

	return kept, attempted


def _sample_high_reward_from_dataset(
	dataset_path: str,
	target_count: int,
	reward_threshold: float,
	seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
	resolved = _resolve_path(dataset_path)
	if not os.path.isfile(resolved):
		raise FileNotFoundError("Dataset not found: %s" % resolved)

	with open(resolved, "rb") as f:
		loaded = pickle.load(f)

	trajectories = _extract_trajectories(loaded)
	high_pool = [t for t in trajectories if _trajectory_return(t) > reward_threshold]

	if len(high_pool) < target_count:
		raise ValueError(
			"SafeDICE high-reward pool too small: %d available with return > %.3f, need %d"
			% (len(high_pool), reward_threshold, target_count)
		)

	rng = np.random.RandomState(seed)
	indices = rng.choice(len(high_pool), size=target_count, replace=False)
	selected = []
	for idx in indices:
		traj = dict(high_pool[int(idx)])
		traj["source"] = np.asarray([1], dtype=np.int32)
		selected.append(traj)

	metadata = {
		"dataset_path": resolved,
		"loaded_trajectory_count": int(len(trajectories)),
		"high_reward_pool_count": int(len(high_pool)),
	}
	return selected, metadata


def _stats(trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
	returns = np.asarray([_trajectory_return(t) for t in trajectories], dtype=np.float32)
	lengths = np.asarray([len(_as_1d(_pick(t, ("rewards", "reward", "rews", "r", "env_rewards")))) for t in trajectories], dtype=np.float32)
	return {
		"count": float(len(trajectories)),
		"return_min": float(np.min(returns)),
		"return_mean": float(np.mean(returns)),
		"return_max": float(np.max(returns)),
		"len_min": float(np.min(lengths)),
		"len_mean": float(np.mean(lengths)),
		"len_max": float(np.max(lengths)),
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build reward-balanced trajectories: noisy low-reward policy rollouts + high-reward SafeDICE samples"
	)
	parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
	parser.add_argument("--safedice_dataset", type=str, default=DEFAULT_SAFEDICE_PATH)
	parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--target_low", type=int, default=900)
	parser.add_argument("--target_high", type=int, default=900)
	parser.add_argument("--reward_threshold", type=float, default=15.0)
	parser.add_argument("--noise_std", type=float, default=0.25)
	parser.add_argument("--max_rollout_episodes", type=int, default=20000)
	parser.add_argument("--max_ep_len", type=int, default=1000)
	parser.add_argument("--deterministic", action="store_true")
	parser.add_argument("--env_id", type=str, default="Safexp-PointGoal1-v0")
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	env, get_action, sess, policy_root, itr = _load_policy(args.policy_path, args.deterministic)
	env = _build_env_if_missing(env, args.env_id)

	print("Policy root: %s" % policy_root)
	print("Policy itr: %s" % str(itr))
	print("Collecting %d low-reward trajectories with return < %.3f using noise_std=%.4f"
		  % (args.target_low, args.reward_threshold, args.noise_std))

	low_trajs, attempted = _collect_noisy_low_reward(
		env=env,
		get_action=get_action,
		target_count=args.target_low,
		reward_threshold=args.reward_threshold,
		noise_std=args.noise_std,
		max_rollout_episodes=args.max_rollout_episodes,
		max_ep_len=args.max_ep_len,
		seed=args.seed,
	)

	print("Sampling %d high-reward trajectories with return > %.3f from SafeDICE dataset"
		  % (args.target_high, args.reward_threshold))
	high_trajs, dataset_meta = _sample_high_reward_from_dataset(
		dataset_path=args.safedice_dataset,
		target_count=args.target_high,
		reward_threshold=args.reward_threshold,
		seed=args.seed,
	)

	combined = low_trajs + high_trajs
	rng = np.random.RandomState(args.seed)
	rng.shuffle(combined)

	out_path = _resolve_path(args.output)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	payload = {
		"trajectories": combined,
		"metadata": {
			"policy_path": _resolve_path(args.policy_path),
			"policy_root": policy_root,
			"policy_itr": str(itr),
			"safedice_dataset": _resolve_path(args.safedice_dataset),
			"reward_threshold": float(args.reward_threshold),
			"target_low": int(args.target_low),
			"target_high": int(args.target_high),
			"noise_std": float(args.noise_std),
			"rollout_attempted_episodes": int(attempted),
			"seed": int(args.seed),
			"dataset_info": dataset_meta,
			"low_stats": _stats(low_trajs),
			"high_stats": _stats(high_trajs),
			"combined_stats": _stats(combined),
		},
	}

	with open(out_path, "wb") as f:
		pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

	print("Saved reward-balanced dataset to %s" % out_path)
	print("Counts: low=%d high=%d combined=%d" % (len(low_trajs), len(high_trajs), len(combined)))

	env.close()
	sess.close()


if __name__ == "__main__":
	main()
