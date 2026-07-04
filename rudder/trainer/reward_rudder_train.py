import argparse
import os

import numpy as np
import torch

try:
    import pickle5 as pickle
except ImportError:
    import pickle

from rudder_train import RudderTrainer


def _repo_root() -> str:
    # File lives in rudder/trainer; repo root is two levels up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(os.path.join(_repo_root(), normalized))


def _default_best_path(save_path: str) -> str:
    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".pt"
    return base + "_best" + ext


def _append_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".pt"
    return base + suffix + ext


def _build_stratified_kfold_indices(labels: np.ndarray, k_folds: int, seed: int):
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    n = len(labels)
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    if k_folds > n:
        raise ValueError("k_folds cannot exceed number of samples")

    idx_pos = np.where(labels == 1.0)[0]
    idx_neg = np.where(labels == 0.0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        raise ValueError("Stratified K-fold requires both classes to be present")
    if min(len(idx_pos), len(idx_neg)) < k_folds:
        raise ValueError(
            "Each class must have at least k_folds samples. "
            "Found class counts: class1=%d class0=%d k_folds=%d"
            % (len(idx_pos), len(idx_neg), k_folds)
        )

    rng = np.random.RandomState(seed)
    idx_pos = idx_pos.copy()
    idx_neg = idx_neg.copy()
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    pos_parts = np.array_split(idx_pos, k_folds)
    neg_parts = np.array_split(idx_neg, k_folds)

    folds = []
    all_idx = np.arange(n)
    for i in range(k_folds):
        val_idx = np.concatenate([pos_parts[i], neg_parts[i]])
        val_idx = val_idx.astype(np.int64, copy=False)
        rng.shuffle(val_idx)
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        folds.append((train_idx, val_idx))

    return folds


def _load_reward_trajectories(dataset_path: str):
    resolved = _resolve_path(dataset_path)
    with open(resolved, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "trajectories" not in payload:
        raise ValueError("Expected dataset dict with key 'trajectories'")

    trajectories = payload["trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("No trajectories found in dataset")

    return trajectories, resolved


def _build_reward_training_tensors(trajectories, reward_threshold: float, seq_len: int = 0):
    first = trajectories[0]
    if "states" not in first or "actions" not in first or "rewards" not in first:
        raise KeyError("Each trajectory must contain states, actions, and rewards")

    state_dim = int(np.asarray(first["states"]).shape[-1])
    action_dim = int(np.asarray(first["actions"]).shape[-1])

    lengths = [
        min(len(t["states"]), len(t["actions"]))
        for t in trajectories
    ]
    target_len = int(max(lengths)) if seq_len <= 0 else int(seq_len)

    feature_dim = state_dim + action_dim
    x = np.zeros((len(trajectories), target_len, feature_dim), dtype=np.float32)
    y = np.zeros((len(trajectories), 1), dtype=np.float32)

    cumulative_rewards = []
    for i, traj in enumerate(trajectories):
        states = np.asarray(traj["states"], dtype=np.float32)
        actions = np.asarray(traj["actions"], dtype=np.float32)
        rewards = np.asarray(traj["rewards"], dtype=np.float32).reshape(-1)

        n = int(min(len(states), len(actions), len(rewards)))
        if n <= 0:
            raise ValueError("Trajectory %d is empty" % i)

        sa = np.concatenate([states[:n], actions[:n]], axis=-1)
        use_n = min(n, target_len)
        x[i, :use_n] = sa[:use_n]

        rsum = float(np.sum(rewards[:n]))
        cumulative_rewards.append(rsum)

        # Label convention requested:
        # 1 -> cumulative reward > threshold (high reward)
        # 0 -> cumulative reward <= threshold (low reward)
        y[i, 0] = 1.0 if rsum > reward_threshold else 0.0

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        state_dim,
        action_dim,
        np.asarray(cumulative_rewards, dtype=np.float32),
        target_len,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reward RUDDER on reward-balanced trajectory dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="rudder/dataset/reward_balanced_1800.pkl",
        help="Path to reward-balanced trajectory dataset",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=15.0,
        help="Threshold for high-reward label",
    )
    parser.add_argument("--seq_len", type=int, default=0, help="Sequence length (0 means max length in data)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler_step_size", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--k_folds", type=int, default=5, help="Number of stratified folds for cross-validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="rudder/models/reward_rudder_cross_val.pt")
    parser.add_argument("--best_save_path", type=str, default="", help="Optional path for best validation checkpoint")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trajectories, resolved_dataset = _load_reward_trajectories(args.dataset_path)
    (
        trajectories_t,
        reward_labels_t,
        state_dim,
        action_dim,
        cumulative_rewards,
        target_len,
    ) = _build_reward_training_tensors(
        trajectories=trajectories,
        reward_threshold=args.reward_threshold,
        seq_len=args.seq_len,
    )

    num_high = int((reward_labels_t.numpy().reshape(-1) == 1.0).sum())
    num_low = int((reward_labels_t.numpy().reshape(-1) == 0.0).sum())

    print("Loaded dataset:", resolved_dataset)
    print("Trajectories: %d" % len(trajectories))
    print("State dim: %d | Action dim: %d | Seq len used: %d" % (state_dim, action_dim, target_len))
    print("Reward threshold: %.3f" % args.reward_threshold)
    print("Label convention: 1 -> reward > threshold, 0 -> reward <= threshold")
    print("Label counts | 1 (reward > %.3f): %d | 0 (reward <= %.3f): %d" % (
        args.reward_threshold,
        num_high,
        args.reward_threshold,
        num_low,
    ))
    print(
        "Cumulative reward stats | min=%.3f avg=%.3f max=%.3f"
        % (float(cumulative_rewards.min()), float(cumulative_rewards.mean()), float(cumulative_rewards.max()))
    )

    print("\n--- Training Reward RUDDER (Stratified %d-Fold CV) ---" % int(args.k_folds))
    labels_np = reward_labels_t.numpy().reshape(-1)
    folds = _build_stratified_kfold_indices(labels_np, int(args.k_folds), int(args.seed))

    fold_results = []
    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        print("\n=== Fold %d/%d ===" % (fold_id, int(args.k_folds)))

        reward_trainer = RudderTrainer(
            state_dim,
            action_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_gamma=args.scheduler_gamma,
            grad_clip_norm=args.grad_clip_norm,
        )
        train_info = reward_trainer.train(
            trajectories_t,
            reward_labels_t,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            seed=args.seed + fold_id,
            train_idx=train_idx,
            val_idx=val_idx,
        )

        fold_results.append(
            {
                "fold": int(fold_id),
                "num_train": int(len(train_idx)),
                "num_val": int(len(val_idx)),
                "final_val_loss": float(train_info["final_val_loss"]),
                "final_val_acc": float(train_info["final_val_acc"]),
                "best_val_loss": float(train_info["best_val_loss"]),
                "best_val_acc": float(train_info["best_val_acc"]),
                "best_epoch": int(train_info["best_epoch"]),
            }
        )

        fold_save_path_raw = _append_suffix(args.save_path, "_fold%d" % fold_id)
        fold_save_path = _resolve_path(fold_save_path_raw)
        os.makedirs(os.path.dirname(fold_save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": reward_trainer.model.state_dict(),
                "baseline": float(reward_trainer.baseline),
                "state_dim": int(state_dim),
                "action_dim": int(action_dim),
                "seq_len": int(target_len),
                "reward_threshold": float(args.reward_threshold),
                "k_folds": int(args.k_folds),
                "fold": int(fold_id),
                "dropout": float(args.dropout),
                "optimizer": "Adam",
                "scheduler": "StepLR",
                "scheduler_step_size": int(args.scheduler_step_size),
                "scheduler_gamma": float(args.scheduler_gamma),
                "grad_clip_norm": float(args.grad_clip_norm),
                "best_epoch": int(train_info["best_epoch"]),
                "best_val_acc": float(train_info["best_val_acc"]),
                "best_val_loss": float(train_info["best_val_loss"]),
                "final_val_acc": float(train_info["final_val_acc"]),
                "final_val_loss": float(train_info["final_val_loss"]),
                "label_convention": "1 if cumulative_reward > threshold else 0",
            },
            fold_save_path,
        )
        print("Saved fold model to: %s" % fold_save_path)

        if reward_trainer.best_state_dict is not None:
            if args.best_save_path:
                best_base = args.best_save_path
            else:
                best_base = _default_best_path(args.save_path)
            fold_best_path_raw = _append_suffix(best_base, "_fold%d" % fold_id)
            fold_best_path = _resolve_path(fold_best_path_raw)
            os.makedirs(os.path.dirname(fold_best_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": reward_trainer.best_state_dict,
                    "baseline": float(reward_trainer.baseline),
                    "state_dim": int(state_dim),
                    "action_dim": int(action_dim),
                    "seq_len": int(target_len),
                    "reward_threshold": float(args.reward_threshold),
                    "k_folds": int(args.k_folds),
                    "fold": int(fold_id),
                    "dropout": float(args.dropout),
                    "optimizer": "Adam",
                    "scheduler": "StepLR",
                    "scheduler_step_size": int(args.scheduler_step_size),
                    "scheduler_gamma": float(args.scheduler_gamma),
                    "grad_clip_norm": float(args.grad_clip_norm),
                    "best_epoch": int(train_info["best_epoch"]),
                    "best_val_acc": float(train_info["best_val_acc"]),
                    "best_val_loss": float(train_info["best_val_loss"]),
                    "final_val_acc": float(train_info["final_val_acc"]),
                    "final_val_loss": float(train_info["final_val_loss"]),
                    "is_best_checkpoint": True,
                    "label_convention": "1 if cumulative_reward > threshold else 0",
                },
                fold_best_path,
            )
            print(
                "Saved BEST fold checkpoint to: %s (epoch=%d val_acc=%.4f val_loss=%.6f)"
                % (
                    fold_best_path,
                    int(train_info["best_epoch"]),
                    float(train_info["best_val_acc"]),
                    float(train_info["best_val_loss"]),
                )
            )

    print("\n=== Stratified %d-Fold Final Validation Metrics ===" % int(args.k_folds))
    for fr in fold_results:
        print(
            "Fold %d | n_train=%d n_val=%d | final_val_acc=%.4f final_val_loss=%.6f"
            % (
                fr["fold"],
                fr["num_train"],
                fr["num_val"],
                fr["final_val_acc"],
                fr["final_val_loss"],
            )
        )

    avg_final_acc = float(np.mean([fr["final_val_acc"] for fr in fold_results]))
    avg_final_loss = float(np.mean([fr["final_val_loss"] for fr in fold_results]))
    print(
        "Average over %d folds | final_val_acc=%.4f final_val_loss=%.6f"
        % (int(args.k_folds), avg_final_acc, avg_final_loss)
    )


if __name__ == "__main__":
    main()
