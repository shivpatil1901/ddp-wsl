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
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
        default="rudder/reward_balanced_1800.pkl",
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="rudder/reward_rudder_new.pt")
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

    print("\n--- Training Reward RUDDER ---")
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
        seed=args.seed,
    )

    save_path = _resolve_path(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": reward_trainer.model.state_dict(),
            "baseline": float(reward_trainer.baseline),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "seq_len": int(target_len),
            "reward_threshold": float(args.reward_threshold),
            "val_split": float(args.val_split),
            "dropout": float(args.dropout),
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "scheduler_step_size": int(args.scheduler_step_size),
            "scheduler_gamma": float(args.scheduler_gamma),
            "grad_clip_norm": float(args.grad_clip_norm),
            "best_epoch": int(train_info["best_epoch"]),
            "best_val_acc": float(train_info["best_val_acc"]),
            "best_val_loss": float(train_info["best_val_loss"]),
            "label_convention": "1 if cumulative_reward > threshold else 0",
        },
        save_path,
    )
    print("Saved trained reward RUDDER to:", save_path)

    if reward_trainer.best_state_dict is not None:
        best_save_path_raw = args.best_save_path if args.best_save_path else _default_best_path(args.save_path)
        best_save_path = _resolve_path(best_save_path_raw)
        os.makedirs(os.path.dirname(best_save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": reward_trainer.best_state_dict,
                "baseline": float(reward_trainer.baseline),
                "state_dim": int(state_dim),
                "action_dim": int(action_dim),
                "seq_len": int(target_len),
                "reward_threshold": float(args.reward_threshold),
                "val_split": float(args.val_split),
                "dropout": float(args.dropout),
                "optimizer": "Adam",
                "scheduler": "StepLR",
                "scheduler_step_size": int(args.scheduler_step_size),
                "scheduler_gamma": float(args.scheduler_gamma),
                "grad_clip_norm": float(args.grad_clip_norm),
                "best_epoch": int(train_info["best_epoch"]),
                "best_val_acc": float(train_info["best_val_acc"]),
                "best_val_loss": float(train_info["best_val_loss"]),
                "is_best_checkpoint": True,
                "label_convention": "1 if cumulative_reward > threshold else 0",
            },
            best_save_path,
        )
        print(
            "Saved BEST reward RUDDER checkpoint to: %s (epoch=%d val_acc=%.4f val_loss=%.6f)"
            % (
                best_save_path,
                int(train_info["best_epoch"]),
                float(train_info["best_val_acc"]),
                float(train_info["best_val_loss"]),
            )
        )


if __name__ == "__main__":
    main()
