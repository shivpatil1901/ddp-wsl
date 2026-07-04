import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
	import pickle5 as pickle  # type: ignore[import-not-found]
except ImportError:
	import pickle

from rudder_train import RudderTrainer


def _repo_root() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)
	if os.path.isabs(normalized):
		return normalized
	return os.path.abspath(os.path.join(_repo_root(), normalized))


def _resolve_input_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)

	candidates = []
	if os.path.isabs(normalized):
		candidates.append(normalized)
	else:
		candidates.append(os.path.abspath(normalized))
		candidates.append(os.path.abspath(os.path.join(_repo_root(), normalized)))
		candidates.append(os.path.abspath(os.path.join(_repo_root(), "rudder", normalized)))
		candidates.append(os.path.abspath(os.path.join(_repo_root(), "rudder", "dataset", normalized)))

	for candidate in candidates:
		if os.path.isfile(candidate):
			return candidate

	raise FileNotFoundError("Input dataset not found: %s. Tried: %s" % (path, ", ".join(candidates)))


def _load_payload(dataset_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	resolved = _resolve_input_path(dataset_path)
	with open(resolved, "rb") as f:
		payload = pickle.load(f)

	if not isinstance(payload, dict) or "trajectories" not in payload:
		raise ValueError("Expected dataset dict with key 'trajectories'")

	trajectories = payload["trajectories"]
	if not isinstance(trajectories, list) or len(trajectories) == 0:
		raise ValueError("No trajectories found in dataset")

	metadata = payload.get("metadata", {})
	if not isinstance(metadata, dict):
		metadata = {}

	return trajectories, metadata, resolved


def _scalar_label(value: Any) -> float:
	arr = np.asarray(value).reshape(-1)
	if arr.size == 0:
		return float("nan")
	return float(arr[0])


def _build_training_tensors(
	trajectories: List[Dict[str, Any]],
	reward_threshold: float,
	cost_threshold: float,
	seq_len: int,
	preferred_source_values: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, int, int, np.ndarray, np.ndarray, int, Dict[str, int]]:
	first = trajectories[0]
	if "states" not in first or "actions" not in first:
		raise KeyError("Each trajectory must contain states and actions")

	state_dim = int(np.asarray(first["states"]).shape[-1])
	action_dim = int(np.asarray(first["actions"]).shape[-1])

	lengths = [min(len(t["states"]), len(t["actions"])) for t in trajectories]
	target_len = int(max(lengths)) if seq_len <= 0 else int(seq_len)

	feature_dim = state_dim + action_dim
	x = np.zeros((len(trajectories), target_len, feature_dim), dtype=np.float32)
	y = np.zeros((len(trajectories), 1), dtype=np.float32)
	cumulative_rewards = np.zeros((len(trajectories),), dtype=np.float32)
	cumulative_costs = np.zeros((len(trajectories),), dtype=np.float32)

	label_source_counts = {
		"preference_label": 0,
		"source": 0,
		"threshold_logic": 0,
	}

	for i, traj in enumerate(trajectories):
		states = np.asarray(traj["states"], dtype=np.float32)
		actions = np.asarray(traj["actions"], dtype=np.float32)
		rewards = np.asarray(traj.get("rewards", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)
		costs = np.asarray(traj.get("costs", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)

		n = int(min(len(states), len(actions), len(rewards), len(costs)))
		if n <= 0:
			raise ValueError(f"Trajectory {i} is empty")

		sa = np.concatenate([states[:n], actions[:n]], axis=-1)
		use_n = min(n, target_len)
		x[i, :use_n] = sa[:use_n]

		c_rew = float(np.sum(rewards[:n]))
		c_cost = float(np.sum(costs[:n]))
		cumulative_rewards[i] = c_rew
		cumulative_costs[i] = c_cost

		label_value = None
		if "preference_label" in traj:
			val = _scalar_label(traj["preference_label"])
			if not np.isnan(val):
				label_value = 1.0 if val > 0.5 else 0.0
				label_source_counts["preference_label"] += 1

		if label_value is None and "source" in traj:
			src = int(round(_scalar_label(traj["source"])))
			label_value = 1.0 if src in preferred_source_values else 0.0
			label_source_counts["source"] += 1

		if label_value is None:
			# Combined-signal fallback label requested by user:
			# preferred (1): high reward AND low cost
			# non-preferred (0): otherwise
			label_value = 1.0 if (c_rew > reward_threshold and c_cost < cost_threshold) else 0.0
			label_source_counts["threshold_logic"] += 1

		y[i, 0] = label_value

	return (
		torch.from_numpy(x),
		torch.from_numpy(y),
		state_dim,
		action_dim,
		cumulative_rewards,
		cumulative_costs,
		target_len,
		label_source_counts,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train RUDDER on combined preference-signal trajectories")
	parser.add_argument(
		"--dataset_path",
		type=str,
		default="rudder/dataset/combined_cost_reward_balanced_1800.pkl",
		help="Path to combined trajectory dataset",
	)
	parser.add_argument("--reward_threshold", type=float, default=15.0)
	parser.add_argument("--cost_threshold", type=float, default=25.0)
	parser.add_argument(
		"--preferred_source_values",
		type=str,
		default="1,3",
		help="Comma-separated source values interpreted as preferred labels",
	)
	parser.add_argument("--seq_len", type=int, default=0, help="Sequence length (0 means max length in data)")
	parser.add_argument("--hidden_dim", type=int, default=64)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--scheduler_step_size", type=int, default=10)
	parser.add_argument("--scheduler_gamma", type=float, default=0.5)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--val_split", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--save_path", type=str, default="rudder/models/reinforce_rudder_combined.pt")
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	preferred_source_values = []
	for token in str(args.preferred_source_values).split(","):
		token = token.strip()
		if token:
			preferred_source_values.append(int(token))

	trajectories, metadata, resolved_dataset = _load_payload(args.dataset_path)
	(
		trajectories_t,
		labels_t,
		state_dim,
		action_dim,
		cumulative_rewards,
		cumulative_costs,
		target_len,
		label_source_counts,
	) = _build_training_tensors(
		trajectories=trajectories,
		reward_threshold=float(args.reward_threshold),
		cost_threshold=float(args.cost_threshold),
		seq_len=int(args.seq_len),
		preferred_source_values=preferred_source_values,
	)

	n_pref = int((labels_t.numpy().reshape(-1) == 1.0).sum())
	n_nonpref = int((labels_t.numpy().reshape(-1) == 0.0).sum())
	if n_pref == 0 or n_nonpref == 0:
		raise ValueError(
			"Need both classes for RUDDER preference training; got preferred=%d non_preferred=%d"
			% (n_pref, n_nonpref)
		)

	print("Loaded dataset:", resolved_dataset)
	print(f"Trajectories: {len(trajectories)}")
	print(f"State dim: {state_dim} | Action dim: {action_dim} | Seq len used: {target_len}")
	print(f"Reward/Cost thresholds: reward>{args.reward_threshold}, cost<{args.cost_threshold}")
	print(f"Label counts | preferred(1): {n_pref} | non_preferred(0): {n_nonpref}")
	print("Label source counts:", label_source_counts)
	print(
		"Cumulative reward stats | min=%.3f avg=%.3f max=%.3f"
		% (float(cumulative_rewards.min()), float(cumulative_rewards.mean()), float(cumulative_rewards.max()))
	)
	print(
		"Cumulative cost stats   | min=%.3f avg=%.3f max=%.3f"
		% (float(cumulative_costs.min()), float(cumulative_costs.mean()), float(cumulative_costs.max()))
	)

	print("\n--- Training Reinforce RUDDER ---")
	trainer = RudderTrainer(
		state_dim,
		action_dim,
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		lr=args.lr,
		scheduler_step_size=args.scheduler_step_size,
		scheduler_gamma=args.scheduler_gamma,
	)
	train_stats = trainer.train(
		trajectories_t,
		labels_t,
		epochs=args.epochs,
		batch_size=args.batch_size,
		val_split=args.val_split,
		seed=args.seed,
	)

	save_path = _resolve_path(args.save_path)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	torch.save(
		{
			"model_state_dict": trainer.model.state_dict(),
			"best_state_dict": trainer.best_state_dict,
			"baseline": float(trainer.baseline),
			"state_dim": int(state_dim),
			"action_dim": int(action_dim),
			"seq_len": int(target_len),
			"reward_threshold": float(args.reward_threshold),
			"cost_threshold": float(args.cost_threshold),
			"preferred_source_values": preferred_source_values,
			"label_source_counts": label_source_counts,
			"dataset_path": resolved_dataset,
			"dataset_metadata": metadata,
			"train_stats": train_stats,
			"val_split": float(args.val_split),
			"dropout": float(args.dropout),
			"optimizer": "Adam",
			"scheduler": "StepLR",
			"scheduler_step_size": int(args.scheduler_step_size),
			"scheduler_gamma": float(args.scheduler_gamma),
			"label_convention": "1 if preferred (explicit label/source or reward>threshold and cost<threshold), else 0",
		},
		save_path,
	)
	print("Saved trained reinforce RUDDER to:", save_path)


if __name__ == "__main__":
	main()
