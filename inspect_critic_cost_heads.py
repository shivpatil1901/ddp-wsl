#!/usr/bin/env python
"""
Inspect SafeDICE checkpoint heads to verify whether reward-like and cost-like signals
can be extracted separately from the loaded model.

This script reports:
- Checkpoint parameter groups (critic_params vs cost_params)
- Inferred dimensions (state_dim, action_dim)
- Output statistics and correlations for:
  - critic(state)
  - cost(state, action)

Usage example:
python inspect_critic_cost_heads.py \
  --weights SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle \
  --dataset SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle \
  --max-states 200000 \
  --batch-size 65536
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from analyze_value_function import ValueFunctionAnalyzer


def _load_pickle(path: str):
    try:
        import pickle5 as pickle_lib
    except ImportError:
        import pickle as pickle_lib

    with open(path, "rb") as f:
        return pickle_lib.load(f)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _infer_dims(training_state: Dict) -> Tuple[int, int]:
    critic_params = training_state.get("critic_params", [])
    cost_params = training_state.get("cost_params", [])

    state_dim = None
    cost_input_dim = None

    for name, param in critic_params:
        if "mlp/dense/kernel" in name or "mlp/dense" in name:
            state_dim = int(param.shape[0])
            break

    for name, param in cost_params:
        if "mlp/dense/kernel" in name or "mlp/dense" in name:
            cost_input_dim = int(param.shape[0])
            break

    if state_dim is None or cost_input_dim is None:
        raise ValueError("Could not infer state/action dimensions from checkpoint parameters")

    action_dim = cost_input_dim - state_dim
    if action_dim <= 0:
        raise ValueError(
            f"Invalid inferred action_dim={action_dim} from state_dim={state_dim}, cost_input_dim={cost_input_dim}"
        )
    return state_dim, action_dim


def _extract_dataset_arrays(data) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(data, dict):
        raise ValueError("Expected dataset to be a dict with arrays (states/observations, actions, rewards, costs)")

    states = data.get("states", data.get("observations", None))
    actions = data.get("actions", None)
    rewards = data.get("rewards", None)
    costs = data.get("costs", None)

    if states is None:
        raise ValueError("Dataset missing both 'states' and 'observations'")

    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(-1, 1)

    if actions is not None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

    if rewards is not None:
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    if costs is not None:
        costs = np.asarray(costs, dtype=np.float32).reshape(-1)

    n = len(states)
    for arr in (actions, rewards, costs):
        if arr is not None:
            n = min(n, len(arr))

    states = states[:n]
    if actions is not None:
        actions = actions[:n]
    if rewards is not None:
        rewards = rewards[:n]
    if costs is not None:
        costs = costs[:n]

    return states, actions, rewards, costs


def _batched_cost_scores(model, states: np.ndarray, actions: np.ndarray, batch_size: int) -> np.ndarray:
    import tensorflow as tf

    n = len(states)
    out = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sa = np.concatenate([states[start:end], actions[start:end]], axis=1)
        sa_tensor = tf.convert_to_tensor(sa, dtype=tf.float32)
        scores, _ = model.cost(sa_tensor)
        out.append(scores.numpy().reshape(-1))
    return np.concatenate(out, axis=0)


def _desc(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr).reshape(-1)
    return (
        f"{name}: mean={arr.mean():.6f}, std={arr.std():.6f}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect SafeDICE critic and cost heads")
    parser.add_argument(
        "--weights",
        type=str,
        default="SafeDICE/weights_LR_HC/antidice_PointGoal1_seed0_20260330_130229_iter400000.pickle",
        help="Path to SafeDICE checkpoint (.pickle)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle",
        help="Dataset containing states/actions/rewards/costs",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=200000,
        help="Max number of states to probe",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Batch size for head inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for optional random sub-sampling",
    )
    parser.add_argument(
        "--random-subsample",
        action="store_true",
        help="If set and max-states < dataset size, sample random indices instead of taking prefix",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="value_function_analysis_combined/critic_cost_head_inspection.txt",
        help="Text report output path",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("Inspecting SafeDICE Critic/Cost Heads")
    print("=" * 88)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    ckpt = _load_pickle(args.weights)
    training_state = ckpt["training_state"]
    critic_params = training_state.get("critic_params", [])
    cost_params = training_state.get("cost_params", [])
    actor_params = training_state.get("actor_params", [])

    state_dim, action_dim = _infer_dims(training_state)

    print(f"Checkpoint: {args.weights}")
    print(f"critic_params tensors: {len(critic_params)}")
    print(f"cost_params tensors:   {len(cost_params)}")
    print(f"actor_params tensors:  {len(actor_params)}")
    print(f"Inferred dims: state_dim={state_dim}, action_dim={action_dim}")

    print("\nLoading analyzer/model...")
    analyzer = ValueFunctionAnalyzer(args.weights)

    print("Loading dataset...")
    data = _load_pickle(args.dataset)
    states, actions, rewards, costs = _extract_dataset_arrays(data)

    n_total = len(states)
    if args.max_states is not None and args.max_states > 0 and args.max_states < n_total:
        if args.random_subsample:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(n_total, size=args.max_states, replace=False)
            states = states[idx]
            if actions is not None:
                actions = actions[idx]
            if rewards is not None:
                rewards = rewards[idx]
            if costs is not None:
                costs = costs[idx]
            print(f"Randomly sub-sampled {len(states)} / {n_total} states (seed={args.seed})")
        else:
            states = states[: args.max_states]
            if actions is not None:
                actions = actions[: args.max_states]
            if rewards is not None:
                rewards = rewards[: args.max_states]
            if costs is not None:
                costs = costs[: args.max_states]
            print(f"Using first {len(states)} / {n_total} states")
    else:
        print(f"Using all {n_total} states")

    if states.shape[1] != state_dim:
        print(
            f"WARNING: Dataset state dim {states.shape[1]} does not match checkpoint inferred state_dim {state_dim}"
        )

    print("\nScoring critic(state)...")
    critic_scores = analyzer.get_state_values(states, batch_size=args.batch_size)
    print(_desc("critic_scores", critic_scores))

    cost_scores = None
    cost_status = "unavailable"
    if actions is None:
        cost_status = "actions missing in dataset"
        print("Skipping cost(state, action): dataset has no actions array")
    elif actions.shape[1] != action_dim:
        cost_status = f"action dim mismatch: dataset={actions.shape[1]}, checkpoint={action_dim}"
        print(f"Skipping cost(state, action): {cost_status}")
    else:
        print("Scoring cost(state, action)...")
        cost_scores = _batched_cost_scores(analyzer.model, states, actions, batch_size=args.batch_size)
        cost_status = "ok"
        print(_desc("cost_scores", cost_scores))

    corr_lines = []
    if rewards is not None:
        c1 = _safe_corr(critic_scores, rewards)
        corr_lines.append(("corr(critic, reward)", c1))
    if costs is not None:
        c2 = _safe_corr(critic_scores, costs)
        corr_lines.append(("corr(critic, cost)", c2))
    if cost_scores is not None and rewards is not None:
        c3 = _safe_corr(cost_scores, rewards)
        corr_lines.append(("corr(cost_head, reward)", c3))
    if cost_scores is not None and costs is not None:
        c4 = _safe_corr(cost_scores, costs)
        corr_lines.append(("corr(cost_head, cost)", c4))

    if cost_scores is not None:
        c5 = _safe_corr(critic_scores, cost_scores)
        corr_lines.append(("corr(critic, cost_head)", c5))

    print("\nCorrelation checks:")
    if corr_lines:
        for name, val in corr_lines:
            print(f"  {name}: {val:.6f}")
    else:
        print("  No reward/cost labels available for correlation checks")

    interpretation = []
    interpretation.append("Separate heads present: YES (critic_params and cost_params both loaded)")
    interpretation.append(f"Cost head usable on this dataset: {cost_status == 'ok'} ({cost_status})")

    if cost_scores is not None and costs is not None:
        c_cost = _safe_corr(cost_scores, costs)
        if np.isnan(c_cost):
            interpretation.append("Cost signal extraction: INCONCLUSIVE (NaN correlation)")
        elif c_cost > 0:
            interpretation.append("Cost signal extraction: PLAUSIBLE (cost_head positively aligned with true costs)")
        else:
            interpretation.append("Cost signal extraction: WEAK/INVERTED (cost_head not positively aligned with true costs)")

    if rewards is not None:
        c_rew = _safe_corr(critic_scores, rewards)
        if np.isnan(c_rew):
            interpretation.append("Reward signal extraction: INCONCLUSIVE (NaN correlation)")
        elif c_rew > 0:
            interpretation.append("Reward signal extraction: PLAUSIBLE (critic positively aligned with rewards)")
        else:
            interpretation.append("Reward signal extraction: WEAK/INVERTED (critic not positively aligned with rewards)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SafeDICE Critic/Cost Head Inspection\n")
        f.write("=" * 88 + "\n\n")
        f.write(f"weights: {args.weights}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"state_dim: {state_dim}\n")
        f.write(f"action_dim: {action_dim}\n")
        f.write(f"states_used: {len(states)}\n")
        f.write(f"critic_params_tensors: {len(critic_params)}\n")
        f.write(f"cost_params_tensors: {len(cost_params)}\n")
        f.write(f"actor_params_tensors: {len(actor_params)}\n")
        f.write(f"cost_head_status: {cost_status}\n\n")
        f.write(_desc("critic_scores", critic_scores) + "\n")
        if cost_scores is not None:
            f.write(_desc("cost_scores", cost_scores) + "\n")
        f.write("\nCorrelation checks:\n")
        for name, val in corr_lines:
            f.write(f"  {name}: {val:.6f}\n")
        f.write("\nInterpretation:\n")
        for line in interpretation:
            f.write(f"  - {line}\n")

    print("\nInterpretation:")
    for line in interpretation:
        print(f"  - {line}")

    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
