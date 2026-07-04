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
DEFAULT_LOW_INPUT = "rudder/dataset/ppo_lagrangian_low_reward_700.pkl"
DEFAULT_TRAIN_COMBINED = "rudder/dataset/reward_balanced_1800.pkl"
DEFAULT_OUTPUT = "rudder/dataset/combined_reward_eval_nonoverlap_200.pkl"


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


def split_flat_dataset_into_trajectories(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    states = pick(payload, ("states", "observations", "obs", "state", "s"))
    actions = pick(payload, ("actions", "acts", "action", "a"))
    next_states = pick(payload, ("next_states", "next_observations", "next_obs", "next_state", "s_next", "obs2"))
    rewards = pick(payload, ("rewards", "reward", "rews", "r", "env_rewards"))
    costs = pick(payload, ("costs", "cost", "c"))
    dones = pick(payload, ("dones", "done", "terminals", "terminal", "episode_ends", "timeouts"))

    if rewards is None:
        raise KeyError("No reward field found in flat dataset")
    if dones is None:
        raise KeyError("No done/terminal field found in flat dataset")

    arrays: Dict[str, np.ndarray] = {}
    if states is not None:
        arrays["states"] = as_np32(states)
    if actions is not None:
        arrays["actions"] = as_np32(actions)
    if next_states is not None:
        arrays["next_states"] = as_np32(next_states)
    arrays["rewards"] = as_np32(rewards)
    if costs is not None:
        arrays["costs"] = as_np32(costs)

    done_arr = as_1d(dones).astype(bool)

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


def extract_trajectories(loaded: Any) -> List[Dict[str, Any]]:
    payload = extract_payload(loaded)
    if isinstance(payload, list):
        trajectories = [t for t in payload if isinstance(t, dict)]
        if not trajectories:
            raise ValueError("Payload list has no trajectory dicts")
        return trajectories
    if isinstance(payload, dict):
        return split_flat_dataset_into_trajectories(payload)
    raise TypeError("Unsupported dataset payload type: %s" % type(payload))


def load_trajectories_from_paths(paths: Iterable[str]) -> List[Dict[str, Any]]:
    all_trajs: List[Dict[str, Any]] = []
    for path in paths:
        resolved = resolve_path(path)
        with open(resolved, "rb") as f:
            loaded = pickle.load(f)
        trajectories = extract_trajectories(loaded)
        all_trajs.extend(trajectories)
        print("Loaded %d trajectories from %s" % (len(trajectories), resolved))
    return all_trajs


def trajectory_reward(traj: Dict[str, Any]) -> float:
    rewards = pick(traj, ("rewards", "reward", "rews", "r", "env_rewards"))
    if rewards is None:
        raise KeyError("Trajectory missing rewards")
    return float(np.sum(as_1d(rewards).astype(np.float32)))


def trajectory_fingerprint(traj: Dict[str, Any]) -> str:
    keys = ("states", "actions", "next_states", "rewards", "costs", "dones")
    h = hashlib.sha1()
    for key in keys:
        if key in traj and traj[key] is not None:
            arr = np.asarray(traj[key], dtype=np.float32)
            h.update(key.encode("utf-8"))
            h.update(str(arr.shape).encode("utf-8"))
            h.update(arr.tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build non-overlapping eval set for rewards: 100 high-reward and 100 low-reward trajectories"
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="High-reward source inputs")
    parser.add_argument("--low_input", default=DEFAULT_LOW_INPUT, help="Low-reward source dataset path")
    parser.add_argument("--train_combined", default=DEFAULT_TRAIN_COMBINED, help="Training dataset to exclude overlap with")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--num_high", type=int, default=100)
    parser.add_argument("--num_low", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    high_source_trajs = load_trajectories_from_paths(args.inputs)
    low_source_trajs = load_trajectories_from_paths([args.low_input])

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
    overlap_high = 0
    overlap_low = 0

    for traj in high_source_trajs:
        fp = trajectory_fingerprint(traj)
        if fp in train_fp:
            overlap_high += 1
            continue
        if trajectory_reward(traj) > args.threshold:
            candidates_high.append(traj)

    for traj in low_source_trajs:
        fp = trajectory_fingerprint(traj)
        if fp in train_fp:
            overlap_low += 1
            continue
        if trajectory_reward(traj) < args.threshold:
            candidates_low.append(traj)

    print("Excluded overlapping high-source trajectories:", overlap_high)
    print("Excluded overlapping low-source trajectories:", overlap_low)
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

    selected_rewards = np.asarray([trajectory_reward(t) for t in selected], dtype=np.float32)
    high_rewards = np.asarray([trajectory_reward(t) for t in selected_high], dtype=np.float32)
    low_rewards = np.asarray([trajectory_reward(t) for t in selected_low], dtype=np.float32)

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
            "low_input": args.low_input,
            "selected_total": int(len(selected)),
            "selected_high_gt_threshold": int((selected_rewards > args.threshold).sum()),
            "selected_low_lt_threshold": int((selected_rewards < args.threshold).sum()),
            "overall_min": float(selected_rewards.min()),
            "overall_mean": float(selected_rewards.mean()),
            "overall_max": float(selected_rewards.max()),
            "high_min": float(high_rewards.min()),
            "high_mean": float(high_rewards.mean()),
            "high_max": float(high_rewards.max()),
            "low_min": float(low_rewards.min()),
            "low_mean": float(low_rewards.mean()),
            "low_max": float(low_rewards.max()),
            "no_overlap_with_train": True,
        },
    }

    out_path = resolve_path(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved non-overlap reward eval dataset to:", out_path)
    print("Summary:")
    for key, value in out_payload["metadata"].items():
        print("  %s: %s" % (key, value))


if __name__ == "__main__":
    main()
