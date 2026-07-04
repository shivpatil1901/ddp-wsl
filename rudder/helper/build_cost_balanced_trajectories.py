"""
Create a cost-balanced trajectory dataset from one or more pickled datasets.

Target composition by default:
- 700 trajectories with cumulative cost >= 25
- 700 trajectories with cumulative cost < 25

Examples:
python rudder/build_cost_balanced_trajectories.py \
  --inputs SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle \
           SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle \
  --output rudder/combined_cost_balanced_1400.pkl
"""

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pickle5 as pickle
except ImportError:
    import pickle


DEFAULT_INPUTS = [
    "SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle",
    "SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle",
]


def _pick(mapping: Dict[str, Any], candidates: Sequence[str]) -> Optional[Any]:
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _as_1d(arr: Any) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def _as_np32(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _resolve_input_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)

    candidates = []
    if os.path.isabs(normalized):
        candidates.append(normalized)
    else:
        candidates.append(os.path.abspath(normalized))
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates.append(os.path.abspath(os.path.join(repo_root, normalized)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Input dataset not found: %s. Tried: %s" % (path, ", ".join(candidates))
    )


def _extract_payload(loaded: Any) -> Any:
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


def _trajectory_cost(traj: Dict[str, Any]) -> float:
    costs = _pick(traj, ("costs", "cost", "c"))
    if costs is None:
        raise KeyError("Trajectory missing costs/cost/c field")
    return float(np.sum(_as_1d(costs).astype(np.float32)))


def _split_flat_dataset_into_trajectories(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    states = _pick(payload, ("states", "observations", "obs", "state", "s"))
    actions = _pick(payload, ("actions", "acts", "action", "a"))
    next_states = _pick(payload, ("next_states", "next_observations", "next_obs", "next_state", "s_next", "obs2"))
    rewards = _pick(payload, ("rewards", "reward", "rews", "r", "env_rewards"))
    costs = _pick(payload, ("costs", "cost", "c"))
    dones = _pick(payload, ("dones", "done", "terminals", "terminal", "episode_ends", "timeouts"))

    if costs is None:
        raise KeyError("No cost field found in flat dataset")
    if dones is None:
        raise KeyError("No done/terminal field found in flat dataset; cannot segment trajectories")

    arrays: Dict[str, np.ndarray] = {}
    if states is not None:
        arrays["states"] = _as_np32(states)
    if actions is not None:
        arrays["actions"] = _as_np32(actions)
    if next_states is not None:
        arrays["next_states"] = _as_np32(next_states)
    if rewards is not None:
        arrays["rewards"] = _as_np32(rewards)
    arrays["costs"] = _as_np32(costs)

    done_arr = _as_1d(dones).astype(bool)

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


def _extract_trajectories(loaded: Any) -> List[Dict[str, Any]]:
    payload = _extract_payload(loaded)

    if isinstance(payload, list):
        trajs = [t for t in payload if isinstance(t, dict)]
        if not trajs:
            raise ValueError("Dataset payload is a list but contains no trajectory dicts")
        return trajs

    if isinstance(payload, dict):
        return _split_flat_dataset_into_trajectories(payload)

    raise TypeError("Unsupported dataset payload type: %s" % type(payload))


def _load_trajectories_from_paths(paths: Iterable[str]) -> List[Dict[str, Any]]:
    all_trajs: List[Dict[str, Any]] = []
    for p in paths:
        resolved = _resolve_input_path(p)
        with open(resolved, "rb") as f:
            loaded = pickle.load(f)
        trajs = _extract_trajectories(loaded)
        all_trajs.extend(trajs)
        print("Loaded %d trajectories from %s" % (len(trajs), resolved))
    return all_trajs


def _sample_balanced(
    trajectories: List[Dict[str, Any]],
    threshold: float,
    num_ge: int,
    num_lt: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ge_pool: List[Dict[str, Any]] = []
    lt_pool: List[Dict[str, Any]] = []

    for traj in trajectories:
        csum = _trajectory_cost(traj)
        if csum >= threshold:
            ge_pool.append(traj)
        else:
            lt_pool.append(traj)

    if len(ge_pool) < num_ge or len(lt_pool) < num_lt:
        raise ValueError(
            "Insufficient trajectories for requested split. "
            "Available >= threshold: %d (need %d), < threshold: %d (need %d)."
            % (len(ge_pool), num_ge, len(lt_pool), num_lt)
        )

    rng = np.random.RandomState(seed)
    ge_idx = rng.choice(len(ge_pool), size=num_ge, replace=False)
    lt_idx = rng.choice(len(lt_pool), size=num_lt, replace=False)

    selected_ge = [ge_pool[int(i)] for i in ge_idx]
    selected_lt = [lt_pool[int(i)] for i in lt_idx]

    combined = selected_ge + selected_lt
    rng.shuffle(combined)

    cost_sums = np.asarray([_trajectory_cost(t) for t in combined], dtype=np.float32)
    ge_cost_sums = np.asarray([_trajectory_cost(t) for t in selected_ge], dtype=np.float32)
    lt_cost_sums = np.asarray([_trajectory_cost(t) for t in selected_lt], dtype=np.float32)

    metadata = {
        "threshold": float(threshold),
        "requested_ge": int(num_ge),
        "requested_lt": int(num_lt),
        "selected_total": int(len(combined)),
        "selected_ge": int(np.sum(cost_sums >= threshold)),
        "selected_lt": int(np.sum(cost_sums < threshold)),
        "cost_sum_mean": float(cost_sums.mean()),
        "cost_sum_std": float(cost_sums.std()),
        "cost_sum_min": float(cost_sums.min()),
        "cost_sum_max": float(cost_sums.max()),
        "ge_cost_sum_mean": float(ge_cost_sums.mean()),
        "ge_cost_sum_std": float(ge_cost_sums.std()),
        "ge_cost_sum_min": float(ge_cost_sums.min()),
        "ge_cost_sum_max": float(ge_cost_sums.max()),
        "lt_cost_sum_mean": float(lt_cost_sums.mean()),
        "lt_cost_sum_std": float(lt_cost_sums.std()),
        "lt_cost_sum_min": float(lt_cost_sums.min()),
        "lt_cost_sum_max": float(lt_cost_sums.max()),
    }
    return combined, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a combined trajectory dataset balanced by cumulative cost"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more input pickle datasets",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="Cumulative cost threshold",
    )
    parser.add_argument(
        "--num_ge",
        type=int,
        default=700,
        help="Number of trajectories with cumulative cost >= threshold",
    )
    parser.add_argument(
        "--num_lt",
        type=int,
        default=700,
        help="Number of trajectories with cumulative cost < threshold",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    trajectories = _load_trajectories_from_paths(args.inputs)
    print("Total trajectory pool: %d" % len(trajectories))

    combined, metadata = _sample_balanced(
        trajectories=trajectories,
        threshold=args.threshold,
        num_ge=args.num_ge,
        num_lt=args.num_lt,
        seed=args.seed,
    )

    out_path = os.path.expanduser(os.path.expandvars(args.output))
    out_path = out_path.replace("\\", os.sep).replace("/", os.sep)
    if not os.path.isabs(out_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_path = os.path.abspath(os.path.join(repo_root, out_path))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "trajectories": combined,
        "metadata": metadata,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved combined dataset to %s" % out_path)
    print("Summary:")
    for k, v in metadata.items():
        print("  %s: %s" % (k, v))


if __name__ == "__main__":
    main()
