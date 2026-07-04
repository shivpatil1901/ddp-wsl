"""
Generate trajectories from a saved policy and keep only low-return trajectories.

Default behavior:
- Loads policy from data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0/simple_save250
- Runs rollouts until 700 trajectories with cumulative reward <= 15 are collected
- Autosaves progress every 100 kept trajectories
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Union

import numpy as np

try:
    import pickle5 as pickle
except ImportError:
    import pickle


DEFAULT_POLICY_PATH = "data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0/simple_save250"
DEFAULT_OUTPUT_PATH = "rudder/ppo_lagrangian_low_reward_700.pkl"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(os.path.join(_repo_root(), normalized))


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


def _compute_stats(trajectories: List[Dict[str, np.ndarray]], attempted: int) -> Dict[str, float]:
    if not trajectories:
        return {
            "num_kept": 0.0,
            "attempted_episodes": float(attempted),
            "reward_min": 0.0,
            "reward_avg": 0.0,
            "reward_max": 0.0,
            "cost_min": 0.0,
            "cost_avg": 0.0,
            "cost_max": 0.0,
            "len_min": 0.0,
            "len_avg": 0.0,
            "len_max": 0.0,
        }

    returns = np.asarray([float(np.sum(t["rewards"])) for t in trajectories], dtype=np.float32)
    costs = np.asarray([float(np.sum(t["costs"])) for t in trajectories], dtype=np.float32)
    lengths = np.asarray([int(len(t["rewards"])) for t in trajectories], dtype=np.float32)

    return {
        "num_kept": float(len(trajectories)),
        "attempted_episodes": float(attempted),
        "reward_min": float(np.min(returns)),
        "reward_avg": float(np.mean(returns)),
        "reward_max": float(np.max(returns)),
        "cost_min": float(np.min(costs)),
        "cost_avg": float(np.mean(costs)),
        "cost_max": float(np.max(costs)),
        "len_min": float(np.min(lengths)),
        "len_avg": float(np.mean(lengths)),
        "len_max": float(np.max(lengths)),
    }


def _save_payload(
    output_path: str,
    trajectories: List[Dict[str, np.ndarray]],
    policy_path: str,
    policy_root: str,
    itr: Union[str, int],
    target_count: int,
    reward_threshold: float,
    attempted: int,
    is_final: bool,
) -> None:
    payload = {
        "trajectories": trajectories,
        "metadata": {
            "policy_path": policy_path,
            "policy_root": policy_root,
            "policy_itr": str(itr),
            "target_count": int(target_count),
            "reward_threshold": float(reward_threshold),
            "is_final": bool(is_final),
            "stats": _compute_stats(trajectories, attempted),
        },
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _collect_low_reward_trajectories(
    env,
    get_action,
    target_count: int,
    reward_threshold: float,
    max_rollout_episodes: int,
    max_ep_len: int,
    output_path: str,
    autosave_every: int,
    policy_path: str,
    policy_root: str,
    itr: Union[str, int],
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, float]]:
    kept: List[Dict[str, np.ndarray]] = []
    attempted = 0
    next_autosave = max(1, autosave_every)

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
            action = get_action(obs)
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
        keep = ep_ret <= reward_threshold
        if keep:
            kept.append(
                {
                    "states": np.asarray(ep_states, dtype=np.float32),
                    "actions": np.asarray(ep_actions, dtype=np.float32),
                    "next_states": np.asarray(ep_next_states, dtype=np.float32),
                    "rewards": np.asarray(ep_rewards, dtype=np.float32),
                    "costs": np.asarray(ep_costs, dtype=np.float32),
                    "dones": np.asarray(ep_dones, dtype=np.float32),
                }
            )

            if len(kept) >= next_autosave:
                _save_payload(
                    output_path=output_path,
                    trajectories=kept,
                    policy_path=policy_path,
                    policy_root=policy_root,
                    itr=itr,
                    target_count=target_count,
                    reward_threshold=reward_threshold,
                    attempted=attempted,
                    is_final=False,
                )
                print("Autosaved progress at kept=%d to %s" % (len(kept), output_path))
                next_autosave += autosave_every

        print(
            "Episode %d | return=%.3f cost=%.3f len=%d | keep=%s | kept=%d/%d"
            % (attempted, ep_ret, ep_cost, ep_len, str(keep), len(kept), target_count)
        )

    if len(kept) < target_count:
        raise RuntimeError(
            "Collected only %d trajectories (target=%d) after %d episodes. "
            "Increase --max_rollout_episodes or relax --reward_threshold."
            % (len(kept), target_count, attempted)
        )

    stats = _compute_stats(kept, attempted)
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 700 low-reward trajectories from a saved policy"
    )
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target_count", type=int, default=700)
    parser.add_argument("--reward_threshold", type=float, default=15.0)
    parser.add_argument("--max_rollout_episodes", type=int, default=10000)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--autosave_every", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--env_id", type=str, default="Safexp-PointGoal1-v0")
    args = parser.parse_args()

    env, get_action, sess, policy_root, itr = _load_policy(args.policy_path, args.deterministic)
    env = _build_env_if_missing(env, args.env_id)

    out_path = _resolve_path(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Policy root: %s" % policy_root)
    print("Policy itr: %s" % str(itr))
    print("Target trajectories: %d" % args.target_count)
    print("Reward threshold (cumulative): <= %.3f" % args.reward_threshold)
    print("Autosave every kept trajectories: %d" % args.autosave_every)

    trajectories, stats = _collect_low_reward_trajectories(
        env=env,
        get_action=get_action,
        target_count=args.target_count,
        reward_threshold=args.reward_threshold,
        max_rollout_episodes=args.max_rollout_episodes,
        max_ep_len=args.max_ep_len,
        output_path=out_path,
        autosave_every=args.autosave_every,
        policy_path=args.policy_path,
        policy_root=policy_root,
        itr=itr,
    )

    _save_payload(
        output_path=out_path,
        trajectories=trajectories,
        policy_path=args.policy_path,
        policy_root=policy_root,
        itr=itr,
        target_count=args.target_count,
        reward_threshold=args.reward_threshold,
        attempted=int(stats["attempted_episodes"]),
        is_final=True,
    )

    print("Saved final dataset: %s" % out_path)
    print("Stats (kept trajectories):")
    print("  reward min/avg/max: %.3f / %.3f / %.3f" % (stats["reward_min"], stats["reward_avg"], stats["reward_max"]))
    print("  cost   min/avg/max: %.3f / %.3f / %.3f" % (stats["cost_min"], stats["cost_avg"], stats["cost_max"]))
    print("  len    min/avg/max: %.0f / %.2f / %.0f" % (stats["len_min"], stats["len_avg"], stats["len_max"]))
    print("  attempted episodes: %d" % int(stats["attempted_episodes"]))

    env.close()
    sess.close()


if __name__ == "__main__":
    main()
