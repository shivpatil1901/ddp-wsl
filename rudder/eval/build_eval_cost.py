import argparse
import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
	import pickle5 as pickle
except ImportError:
	import pickle


DEFAULT_INPUTS = [
	"SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle",
	"SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle",
]
DEFAULT_TRAIN_COMBINED = "rudder/combined_cost_balanced_1400.pkl"
DEFAULT_OUTPUT = "rudder/eval/combined_cost_eval_nonoverlap_200.pkl"


def pick(mapping: Dict[str, Any], candidates: Sequence[str]) -> Optional[Any]:
	for key in candidates:
		if key in mapping:
			return mapping[key]
	return None


def as_1d(arr: Any) -> np.ndarray:
	return np.asarray(arr).reshape(-1)


def as_np32(arr: Any) -> np.ndarray:
	return np.asarray(arr, dtype=np.float32)


def repo_root() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)
	if os.path.isabs(normalized):
		return normalized
	return os.path.abspath(os.path.join(repo_root(), normalized))


def extract_payload(loaded: Any) -> Any:
	payload = loaded
	if isinstance(payload, dict):
		ts = payload.get("training_state")
		if isinstance(ts, dict):
			for k in ("dataset", "data", "replay_buffer", "buffer", "expert_data", "trajectories"):
				if k in ts:
					payload = ts[k]
					break
		if payload is loaded:
			for k in ("dataset", "data", "expert_data", "trajectories"):
				if k in payload:
					payload = payload[k]
					break
	return payload


def split_flat_dataset_into_trajectories(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
	states = pick(payload, ("states", "observations", "obs", "state", "s"))
	actions = pick(payload, ("actions", "acts", "action", "a"))
	next_states = pick(payload, ("next_states", "next_observations", "next_obs", "next_state", "s_next", "obs2"))
	rewards = pick(payload, ("rewards", "reward", "rews", "r", "env_rewards"))
	costs = pick(payload, ("costs", "cost", "c"))
	dones = pick(payload, ("dones", "done", "terminals", "terminal", "episode_ends", "timeouts"))

	if costs is None:
		raise KeyError("No cost field found in flat dataset")
	if dones is None:
		raise KeyError("No done/terminal field found in flat dataset")

	arrays: Dict[str, np.ndarray] = {}
	if states is not None:
		arrays["states"] = as_np32(states)
	if actions is not None:
		arrays["actions"] = as_np32(actions)
	if next_states is not None:
		arrays["next_states"] = as_np32(next_states)
	if rewards is not None:
		arrays["rewards"] = as_np32(rewards)
	arrays["costs"] = as_np32(costs)

	done_arr = as_1d(dones).astype(bool)

	lengths = [len(done_arr)] + [len(v) for v in arrays.values()]
	n = int(min(lengths))
	done_arr = done_arr[:n]
	for k in list(arrays.keys()):
		arrays[k] = arrays[k][:n]

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


def extract_trajectories(loaded: Any) -> List[Dict[str, Any]]:
	payload = extract_payload(loaded)
	if isinstance(payload, list):
		trajs = [t for t in payload if isinstance(t, dict)]
		if not trajs:
			raise ValueError("Payload list has no trajectory dicts")
		return trajs
	if isinstance(payload, dict):
		return split_flat_dataset_into_trajectories(payload)
	raise TypeError("Unsupported dataset payload type: %s" % type(payload))


def load_trajectories_from_paths(paths: Iterable[str]) -> List[Dict[str, Any]]:
	all_trajs: List[Dict[str, Any]] = []
	for p in paths:
		rp = resolve_path(p)
		with open(rp, "rb") as f:
			loaded = pickle.load(f)
		trajs = extract_trajectories(loaded)
		all_trajs.extend(trajs)
		print("Loaded %d trajectories from %s" % (len(trajs), rp))
	return all_trajs


def trajectory_cost(traj: Dict[str, Any]) -> float:
	costs = pick(traj, ("costs", "cost", "c"))
	if costs is None:
		raise KeyError("Trajectory missing costs")
	return float(np.sum(as_1d(costs).astype(np.float32)))


def trajectory_fingerprint(traj: Dict[str, Any]) -> str:
	keys = ("states", "actions", "next_states", "rewards", "costs", "dones")
	h = hashlib.sha1()
	for k in keys:
		if k in traj and traj[k] is not None:
			arr = np.asarray(traj[k], dtype=np.float32)
			h.update(k.encode("utf-8"))
			h.update(str(arr.shape).encode("utf-8"))
			h.update(arr.tobytes())
	return h.hexdigest()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build non-overlapping eval set: 100 high-cost and 100 low-cost trajectories"
	)
	parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
	parser.add_argument("--train_combined", default=DEFAULT_TRAIN_COMBINED)
	parser.add_argument("--output", default=DEFAULT_OUTPUT)
	parser.add_argument("--threshold", type=float, default=25.0)
	parser.add_argument("--num_high", type=int, default=100)
	parser.add_argument("--num_low", type=int, default=100)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	source_trajs = load_trajectories_from_paths(args.inputs)

	train_path = resolve_path(args.train_combined)
	with open(train_path, "rb") as f:
		train_payload = pickle.load(f)
	if not isinstance(train_payload, dict) or "trajectories" not in train_payload:
		raise ValueError("train_combined must contain key trajectories")
	train_trajs = train_payload["trajectories"]

	train_fp = set(trajectory_fingerprint(t) for t in train_trajs)
	print("Train set fingerprints:", len(train_fp))

	candidates_high = []
	candidates_low = []
	overlap_count = 0

	for t in source_trajs:
		fp = trajectory_fingerprint(t)
		if fp in train_fp:
			overlap_count += 1
			continue

		csum = trajectory_cost(t)
		if csum > args.threshold:
			candidates_high.append(t)
		elif csum < args.threshold:
			candidates_low.append(t)

	print("Excluded overlapping trajectories:", overlap_count)
	print("Non-overlap high pool (> %.1f): %d" % (args.threshold, len(candidates_high)))
	print("Non-overlap low pool  (< %.1f): %d" % (args.threshold, len(candidates_low)))

	if len(candidates_high) < args.num_high or len(candidates_low) < args.num_low:
		raise ValueError(
			"Not enough non-overlap candidates. Need high=%d low=%d, got high=%d low=%d"
			% (args.num_high, args.num_low, len(candidates_high), len(candidates_low))
		)

	rng = np.random.RandomState(args.seed)
	hi_idx = rng.choice(len(candidates_high), size=args.num_high, replace=False)
	lo_idx = rng.choice(len(candidates_low), size=args.num_low, replace=False)

	selected_high = [candidates_high[int(i)] for i in hi_idx]
	selected_low = [candidates_low[int(i)] for i in lo_idx]
	selected = selected_high + selected_low
	rng.shuffle(selected)

	selected_costs = np.asarray([trajectory_cost(t) for t in selected], dtype=np.float32)
	high_costs = np.asarray([trajectory_cost(t) for t in selected_high], dtype=np.float32)
	low_costs = np.asarray([trajectory_cost(t) for t in selected_low], dtype=np.float32)

	selected_fp = set(trajectory_fingerprint(t) for t in selected)
	assert len(selected_fp.intersection(train_fp)) == 0, "Overlap detected unexpectedly"

	out_payload = {
		"trajectories": selected,
		"metadata": {
			"threshold": float(args.threshold),
			"num_high": int(args.num_high),
			"num_low": int(args.num_low),
			"train_combined": args.train_combined,
			"inputs": args.inputs,
			"selected_total": int(len(selected)),
			"selected_high_gt_threshold": int((selected_costs > args.threshold).sum()),
			"selected_low_lt_threshold": int((selected_costs < args.threshold).sum()),
			"overall_min": float(selected_costs.min()),
			"overall_mean": float(selected_costs.mean()),
			"overall_max": float(selected_costs.max()),
			"high_min": float(high_costs.min()),
			"high_mean": float(high_costs.mean()),
			"high_max": float(high_costs.max()),
			"low_min": float(low_costs.min()),
			"low_mean": float(low_costs.mean()),
			"low_max": float(low_costs.max()),
			"no_overlap_with_train": True,
		},
	}

	out_path = resolve_path(args.output)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "wb") as f:
		pickle.dump(out_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

	print("Saved non-overlap eval dataset to:", out_path)
	print("Summary:")
	for k, v in out_payload["metadata"].items():
		print("  %s: %s" % (k, v))


if __name__ == "__main__":
	main()
