import os
import numpy as np
import torch
import torch.nn as nn

try:
    import pickle5 as pickle
except ImportError:
    import pickle


# -----------------------------
# Paths
# -----------------------------
repo_root = "/home/shiv1901/safeil-data-collection-main/safeil-data-collection-main"
model_path = os.path.join(repo_root, "rudder", "models", "cost_rudder_1.pt")
data_path = os.path.join(repo_root, "rudder", "combined_cost_balanced_1400.pkl")
eval_data_path = os.path.join(repo_root, "rudder", "eval", "combined_cost_eval_nonoverlap_200.pkl")


# -----------------------------
# Model definitions
# -----------------------------
class OriginalRUDDER(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.output_layer(lstm_out)


class LayerNormRUDDER(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        norm_out = self.layer_norm(lstm_out)
        drop_out = self.dropout(norm_out)
        return self.output_layer(drop_out)


def _build_model_from_ckpt(state_dim: int, action_dim: int, hidden_dim: int, ckpt_state: dict) -> nn.Module:
    has_layer_norm = "layer_norm.weight" in ckpt_state and "layer_norm.bias" in ckpt_state
    if has_layer_norm:
        print("Detected checkpoint architecture: LayerNormRUDDER")
        return LayerNormRUDDER(state_dim, action_dim, hidden_dim=hidden_dim, dropout=0.2)

    print("Detected checkpoint architecture: OriginalRUDDER")
    return OriginalRUDDER(state_dim, action_dim, hidden_dim=hidden_dim)

# -----------------------------
# Load model checkpoint
# -----------------------------
ckpt = torch.load(model_path, map_location="cpu")
state_dim = int(ckpt["state_dim"])
action_dim = int(ckpt["action_dim"])
seq_len = int(ckpt["seq_len"])
threshold = float(ckpt["cost_threshold"])
baseline = float(ckpt.get("baseline", 0.0))

# infer hidden size from LSTM weight shape: (4*hidden_dim, input_dim)
hidden_dim = ckpt["model_state_dict"]["lstm.weight_ih_l0"].shape[0] // 4
print(f"Inferred hidden_dim from checkpoint: {hidden_dim}")
model = _build_model_from_ckpt(state_dim, action_dim, hidden_dim, ckpt["model_state_dict"])
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()


def evaluate_dataset(dataset_path: str, split_name: str) -> None:
    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    trajectories = payload["trajectories"]

    # Build same tensor format used in training.
    n_traj = len(trajectories)
    x = np.zeros((n_traj, seq_len, state_dim + action_dim), dtype=np.float32)
    y_true = np.zeros((n_traj, 1), dtype=np.float32)
    cost_sums = np.zeros((n_traj,), dtype=np.float32)

    for i, traj in enumerate(trajectories):
        s = np.asarray(traj["states"], dtype=np.float32)
        a = np.asarray(traj["actions"], dtype=np.float32)
        c = np.asarray(traj["costs"], dtype=np.float32).reshape(-1)

        n = min(len(s), len(a), len(c))
        sa = np.concatenate([s[:n], a[:n]], axis=-1)
        use_n = min(n, seq_len)
        x[i, :use_n] = sa[:use_n]

        csum = float(np.sum(c[:n]))
        cost_sums[i] = csum
        y_true[i, 0] = 1.0 if csum < threshold else 0.0

    x_t = torch.from_numpy(x)
    y_true_t = torch.from_numpy(y_true)

    with torch.no_grad():
        all_preds = model(x_t)
        final_pred = all_preds[:, -1, :]
        pred_label = final_pred + baseline
        pred_prob_like = torch.sigmoid(pred_label)
        pred_class = (pred_label >= 0.5).float()

    acc = (pred_class == y_true_t).float().mean().item()
    mse = torch.mean((pred_label - y_true_t) ** 2).item()

    # Precision/Recall for positive class (label 1: cumulative cost < threshold)
    tp = ((pred_class == 1.0) & (y_true_t == 1.0)).sum().item()
    fp = ((pred_class == 1.0) & (y_true_t == 0.0)).sum().item()
    fn = ((pred_class == 0.0) & (y_true_t == 1.0)).sum().item()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    p = pred_label.squeeze(-1).cpu().numpy()
    t = y_true.squeeze(-1)

    print(f"=== {split_name} Cost Prediction Summary ===")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"N trajectories: {n_traj}")
    print(f"Label rule: 1 if cumulative cost < {threshold}, else 0")
    print(f"Baseline used during training: {baseline:.6f}")
    print(f"MSE (on reconstructed labels): {mse:.6f}")
    print(f"Accuracy (threshold 0.5): {acc:.4f}")
    print(f"Precision (class 1): {precision:.4f}")
    print(f"Recall (class 1): {recall:.4f}")
    print(f"Pred label stats min/mean/max: {p.min():.4f} / {p.mean():.4f} / {p.max():.4f}")
    print(f"True label stats  min/mean/max: {t.min():.1f} / {t.mean():.4f} / {t.max():.1f}")

    print("\nFirst 20 samples:")
    for i in range(min(20, n_traj)):
        print(
            f"idx={i:4d}  cost_sum={cost_sums[i]:7.2f}  true={int(y_true[i,0])}  "
            f"pred={float(pred_label[i,0]):7.4f}  sigmoid={float(pred_prob_like[i,0]):7.4f}  "
            f"class={int(pred_class[i,0].item())}"
        )


evaluate_dataset(data_path, "Train-set")
print()
evaluate_dataset(eval_data_path, "Eval-set (non-overlap)")
