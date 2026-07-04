"""
Build a combined trajectory dataset for RUDDER training with two equal pools:
- 900 high-reward, low-cost trajectories from SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle
- 900 low-reward, high-cost trajectories from SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle

Selection criteria:
- Preferred pool: cumulative reward > 15 and cumulative cost < 25
- Non-preferred pool: cumulative reward < 15 and cumulative cost > 25

The output is saved in the same trajectory-list format used by
build_reward_balanced_trajectories.py so it can be consumed by the RUDDER
training class in rudder/trainer/rudder_train.py.
"""

import argparse
import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pickle5 as pickle  # type: ignore[import-not-found]
except ImportError:
    import pickle


DEFAULT_PREFERRED_DATASET = "SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
DEFAULT_NON_PREFERRED_DATASET = "SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle"
DEFAULT_PREFERRED_POLICY = "data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0/simple_save332"
DEFAULT_OUTPUT_PATH = "rudder/dataset/combined_cost_reward_balanced_1800.pkl"
DEFAULT_REWARD_THRESHOLD = 15.0
DEFAULT_COST_THRESHOLD = 25.0
DEFAULT_TARGET_PER_CLASS = 900


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "SafeDICE", "dataset", "safetygym", normalized)))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "SafeDICE", "dataset", "safetygym_original", normalized)))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "SafeDICE", "dataset", normalized)))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "SafeDICE", normalized)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError("Input dataset not found: %s. Tried: %s" % (path, ", ".join(candidates)))


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


def _load_pickle_like(path: str) -> Any:
    with open(path, "rb") as f:
        header = f.read(256)

    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "Dataset file appears to be a Git LFS pointer, not actual pickle data: %s. "
            "Run 'git lfs pull' (or fetch this dataset) and retry."
            % path
        )

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ValueError as exc:
        msg = str(exc)
        if "unsupported pickle protocol: 5" in msg:
            raise RuntimeError(
                "Failed to load protocol-5 pickle at %s using interpreter %s and module %s. "
                "Install pickle5 in this env (pip install pickle5) or run with Python >= 3.8."
                % (path, sys.version.split()[0], pickle.__name__)
            )
        raise


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

    from safe_rl.utils.load_utils import load_policy  # type: ignore[import-not-found]

    policy_root, itr = _policy_root_and_itr(policy_path)
    env, get_action, sess = load_policy(policy_root, itr=itr, deterministic=deterministic)
    return env, get_action, sess, policy_root, itr


def _build_env_if_missing(env, env_id: str):
    if env is not None:
        return env

    import gym
    try:
        importlib.import_module("safety_gym")
    except ImportError:
        pass

    return gym.make(env_id)


def _trajectory_stats(traj: Dict[str, Any]) -> Dict[str, float]:
    rewards = _pick(traj, ("rewards", "reward", "rews", "r", "env_rewards"))
    costs = _pick(traj, ("costs", "cost", "c"))
    if rewards is None:
        raise KeyError("Trajectory missing rewards/reward/rews/r field")

    reward_arr = _as_1d(rewards).astype(np.float32)
    cost_arr = _as_1d(costs).astype(np.float32) if costs is not None else np.zeros_like(reward_arr)

    return {
        "cum_reward": float(np.sum(reward_arr)),
        "cum_cost": float(np.sum(cost_arr)),
        "length": float(len(reward_arr)),
    }


def _compute_stats(trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trajectories:
        return {
            "count": 0.0,
            "reward_min": float("nan"),
            "reward_mean": float("nan"),
            "reward_std": float("nan"),
            "reward_max": float("nan"),
            "cost_min": float("nan"),
            "cost_mean": float("nan"),
            "cost_std": float("nan"),
            "cost_max": float("nan"),
        }

    stats = [_trajectory_stats(t) for t in trajectories]
    rewards = np.asarray([s["cum_reward"] for s in stats], dtype=np.float32)
    costs = np.asarray([s["cum_cost"] for s in stats], dtype=np.float32)
    lengths = np.asarray([s["length"] for s in stats], dtype=np.float32)

    return {
        "count": float(len(trajectories)),
        "reward_min": float(np.min(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_max": float(np.max(rewards)),
        "cost_min": float(np.min(costs)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "cost_max": float(np.max(costs)),
        "len_min": float(np.min(lengths)),
        "len_mean": float(np.mean(lengths)),
        "len_max": float(np.max(lengths)),
    }


def _filter_pool(
    trajectories: List[Dict[str, Any]],
    reward_threshold: float,
    cost_threshold: float,
    reward_op: str,
    cost_op: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for traj in trajectories:
        stats = _trajectory_stats(traj)
        reward_ok = stats["cum_reward"] > reward_threshold if reward_op == ">" else stats["cum_reward"] < reward_threshold
        cost_ok = stats["cum_cost"] < cost_threshold if cost_op == "<" else stats["cum_cost"] > cost_threshold
        if reward_ok and cost_ok:
            filtered.append(traj)
    return filtered


def _select_n(trajectories: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    if len(trajectories) < count:
        raise ValueError("Pool too small: %d available, need %d" % (len(trajectories), count))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(trajectories), size=count, replace=False)
    selected: List[Dict[str, Any]] = []
    for idx in indices:
        selected.append(dict(trajectories[int(idx)]))
    return selected


def _collect_preferred_high_reward_low_cost(
    env,
    get_action,
    target_count: int,
    reward_threshold: float,
    cost_threshold: float,
    noise_std: float,
    noise_max_std: float,
    noise_growth: float,
    noise_patience: int,
    random_action_prob: float,
    max_rollout_episodes: int,
    max_ep_len: int,
    seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    rng = np.random.RandomState(seed)
    kept: List[Dict[str, np.ndarray]] = []
    attempted = 0
    current_noise_std = float(noise_std)
    no_keep_streak = 0

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
            if current_noise_std > 0.0:
                action = action + rng.normal(loc=0.0, scale=current_noise_std, size=action.shape).astype(np.float32)

            if random_action_prob > 0.0 and hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                if float(rng.rand()) < float(random_action_prob):
                    action = rng.uniform(low=env.action_space.low, high=env.action_space.high).astype(np.float32)

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
        keep = (ep_ret > reward_threshold) and (ep_cost < cost_threshold)
        if keep:
            no_keep_streak = 0
            kept.append(
                {
                    "states": np.asarray(ep_states, dtype=np.float32),
                    "actions": np.asarray(ep_actions, dtype=np.float32),
                    "next_states": np.asarray(ep_next_states, dtype=np.float32),
                    "rewards": np.asarray(ep_rewards, dtype=np.float32),
                    "costs": np.asarray(ep_costs, dtype=np.float32),
                    "dones": np.asarray(ep_dones, dtype=np.float32),
                    "source": np.asarray([2], dtype=np.int32),
                }
            )
        else:
            no_keep_streak += 1
            if no_keep_streak >= max(1, int(noise_patience)):
                new_noise = min(float(noise_max_std), float(current_noise_std) * float(noise_growth))
                if new_noise > current_noise_std:
                    current_noise_std = new_noise
                    print(
                        "Increasing preferred-rollout noise_std to %.4f after %d consecutive non-kept rollouts"
                        % (current_noise_std, no_keep_streak)
                    )
                no_keep_streak = 0

        print(
            "Preferred rollout %d | return=%.3f cost=%.3f len=%d | keep(>%.2f and <%.2f)=%s | kept=%d/%d | noise_std=%.4f"
            % (
                attempted,
                ep_ret,
                ep_cost,
                ep_len,
                reward_threshold,
                cost_threshold,
                str(keep),
                len(kept),
                target_count,
                current_noise_std,
            )
        )

    if len(kept) < target_count:
        raise RuntimeError(
            "Collected only %d preferred trajectories (target=%d) after %d episodes. "
            "Increase --preferred_max_rollout_episodes, increase --preferred_noise_std/--preferred_noise_max_std, set --preferred_random_action_prob, or relax thresholds."
            % (len(kept), target_count, attempted)
        )

    return kept, attempted


def _default_output_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".pkl"
    return base + "_combined" + ext


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a combined 1800-trajectory dataset for RUDDER training"
    )
    parser.add_argument(
        "--preferred_dataset",
        type=str,
        default=DEFAULT_PREFERRED_DATASET,
        help="SafeDICE dataset for preferred pool (high reward, low cost)",
    )
    parser.add_argument(
        "--non_preferred_dataset",
        type=str,
        default=DEFAULT_NON_PREFERRED_DATASET,
        help="SafeDICE dataset for non-preferred pool (low reward, high cost)",
    )
    parser.add_argument(
        "--preferred_policy_path",
        type=str,
        default=DEFAULT_PREFERRED_POLICY,
        help="PPO-Lagrangian policy path used to generate extra preferred rollouts if needed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output pickle path",
    )
    parser.add_argument("--target_high", type=int, default=DEFAULT_TARGET_PER_CLASS)
    parser.add_argument("--target_low", type=int, default=DEFAULT_TARGET_PER_CLASS)
    parser.add_argument("--reward_threshold_high", type=float, default=DEFAULT_REWARD_THRESHOLD)
    parser.add_argument("--reward_threshold_low", type=float, default=DEFAULT_REWARD_THRESHOLD)
    parser.add_argument("--cost_threshold_low", type=float, default=DEFAULT_COST_THRESHOLD)
    parser.add_argument("--cost_threshold_high", type=float, default=DEFAULT_COST_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preferred_noise_std", type=float, default=0.15)
    parser.add_argument("--preferred_noise_max_std", type=float, default=0.75)
    parser.add_argument("--preferred_noise_growth", type=float, default=1.2)
    parser.add_argument("--preferred_noise_patience", type=int, default=50)
    parser.add_argument("--preferred_random_action_prob", type=float, default=0.05)
    parser.add_argument("--preferred_max_rollout_episodes", type=int, default=20000)
    parser.add_argument("--preferred_max_ep_len", type=int, default=1000)
    parser.add_argument("--preferred_deterministic", action="store_true")
    parser.add_argument("--env_id", type=str, default="Safexp-PointGoal1-v0")
    args = parser.parse_args()

    preferred_path = _resolve_input_path(args.preferred_dataset)
    non_preferred_path = _resolve_input_path(args.non_preferred_dataset)

    print("Loading preferred dataset:", preferred_path)
    preferred_loaded = _load_pickle_like(preferred_path)
    preferred_trajs = _extract_trajectories(preferred_loaded)
    print("Preferred trajectories available:", len(preferred_trajs))

    print("Loading non-preferred dataset:", non_preferred_path)
    non_preferred_loaded = _load_pickle_like(non_preferred_path)
    non_preferred_trajs = _extract_trajectories(non_preferred_loaded)
    print("Non-preferred trajectories available:", len(non_preferred_trajs))

    preferred_pool = _filter_pool(
        preferred_trajs,
        reward_threshold=float(args.reward_threshold_high),
        cost_threshold=float(args.cost_threshold_low),
        reward_op=">",
        cost_op="<",
    )
    non_preferred_pool = _filter_pool(
        non_preferred_trajs,
        reward_threshold=float(args.reward_threshold_low),
        cost_threshold=float(args.cost_threshold_high),
        reward_op="<",
        cost_op=">",
    )

    print(
        "Preferred pool criteria: cum_reward > %.3f and cum_cost < %.3f -> %d trajectories"
        % (args.reward_threshold_high, args.cost_threshold_low, len(preferred_pool))
    )
    print(
        "Non-preferred pool criteria: cum_reward < %.3f and cum_cost > %.3f -> %d trajectories"
        % (args.reward_threshold_low, args.cost_threshold_high, len(non_preferred_pool))
    )

    preferred_initial_count = min(len(preferred_pool), int(args.target_high))
    non_preferred_initial_count = min(len(non_preferred_pool), int(args.target_low))
    preferred_selected = _select_n(preferred_pool, preferred_initial_count, seed=int(args.seed)) if preferred_initial_count > 0 else []
    non_preferred_selected = _select_n(non_preferred_pool, non_preferred_initial_count, seed=int(args.seed) + 1) if non_preferred_initial_count > 0 else []

    preferred_attempted = 0
    if len(preferred_selected) < int(args.target_high):
        needed = int(args.target_high) - len(preferred_selected)
        print(
            "Preferred pool short by %d trajectories; generating rollouts from policy %s"
            % (needed, args.preferred_policy_path)
        )
        env = None
        sess = None
        get_action = None
        try:
            env, get_action, sess, policy_root, itr = _load_policy(args.preferred_policy_path, args.preferred_deterministic)
            env = _build_env_if_missing(env, args.env_id)
            generated, preferred_attempted = _collect_preferred_high_reward_low_cost(
                env=env,
                get_action=get_action,
                target_count=needed,
                reward_threshold=float(args.reward_threshold_high),
                cost_threshold=float(args.cost_threshold_low),
                noise_std=float(args.preferred_noise_std),
                noise_max_std=float(args.preferred_noise_max_std),
                noise_growth=float(args.preferred_noise_growth),
                noise_patience=int(args.preferred_noise_patience),
                random_action_prob=float(args.preferred_random_action_prob),
                max_rollout_episodes=int(args.preferred_max_rollout_episodes),
                max_ep_len=int(args.preferred_max_ep_len),
                seed=int(args.seed),
            )
            for traj in generated:
                traj["source"] = np.asarray([3], dtype=np.int32)
                traj["preference_label"] = np.asarray([1], dtype=np.int32)
            preferred_selected.extend(generated)
            print(
                "Top-up preferred trajectories generated: %d (policy_root=%s itr=%s)"
                % (len(generated), policy_root, str(itr))
            )
        finally:
            if env is not None:
                env.close()
            if sess is not None:
                sess.close()

    for traj in preferred_selected:
        traj["source"] = np.asarray([1], dtype=np.int32)
        traj["preference_label"] = np.asarray([1], dtype=np.int32)
    for traj in non_preferred_selected:
        traj["source"] = np.asarray([0], dtype=np.int32)
        traj["preference_label"] = np.asarray([0], dtype=np.int32)

    combined = preferred_selected + non_preferred_selected
    rng = np.random.RandomState(int(args.seed))
    rng.shuffle(combined)

    out_path = _resolve_path(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "trajectories": combined,
        "metadata": {
            "preferred_dataset": preferred_path,
            "non_preferred_dataset": non_preferred_path,
            "reward_threshold_high": float(args.reward_threshold_high),
            "reward_threshold_low": float(args.reward_threshold_low),
            "cost_threshold_low": float(args.cost_threshold_low),
            "cost_threshold_high": float(args.cost_threshold_high),
            "target_high": int(args.target_high),
            "target_low": int(args.target_low),
            "seed": int(args.seed),
            "preferred_pool_size": int(len(preferred_pool)),
            "non_preferred_pool_size": int(len(non_preferred_pool)),
            "preferred_stats": _compute_stats(preferred_selected),
            "non_preferred_stats": _compute_stats(non_preferred_selected),
            "combined_stats": _compute_stats(combined),
            "label_convention": {
                "preferred": "cum_reward > reward_threshold_high and cum_cost < cost_threshold_low",
                "non_preferred": "cum_reward < reward_threshold_low and cum_cost > cost_threshold_high",
            },
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved combined RUDDER dataset to:", out_path)
    print("Counts: preferred=%d non_preferred=%d combined=%d" % (len(preferred_selected), len(non_preferred_selected), len(combined)))
    print("Combined reward stats:", payload["metadata"]["combined_stats"])


if __name__ == "__main__":
    main()
