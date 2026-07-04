import os
import numpy as np
import torch
import torch.nn as nn

try:
    import pickle5 as pickle
except ImportError:
    import pickle


# =========================
# Config
# =========================
repo_root = "/home/shiv1901/safeil-data-collection-main/safeil-data-collection-main"
model_path = os.path.join(repo_root, "rudder", "models", "reward_rudder_1_best.pt")
data_path = os.path.join(repo_root, "rudder", "dataset", "reward_balanced_1800.pkl")
positive_reward_threshold = 0.0  # reward > 0 treated as positive-reward step


# =========================
# Model
# =========================
class OriginalRUDDER(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.output_layer(y)


class LayerNormRUDDER(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        y, _ = self.lstm(x)
        y = self.layer_norm(y)
        y = self.dropout(y)
        return self.output_layer(y)


def build_model_from_ckpt(ckpt):
    hidden_dim = ckpt["model_state_dict"]["lstm.weight_ih_l0"].shape[0] // 4
    has_layer_norm = "layer_norm.weight" in ckpt["model_state_dict"]

    if has_layer_norm:
        print("Detected checkpoint architecture: LayerNormRUDDER")
        model = LayerNormRUDDER(
            state_dim=int(ckpt["state_dim"]),
            action_dim=int(ckpt["action_dim"]),
            hidden_dim=hidden_dim,
            dropout=float(ckpt.get("dropout", 0.2)),
        )
    else:
        print("Detected checkpoint architecture: OriginalRUDDER")
        model = OriginalRUDDER(
            state_dim=int(ckpt["state_dim"]),
            action_dim=int(ckpt["action_dim"]),
            hidden_dim=hidden_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def redistributed_from_model(model, states, actions, baseline=0.0):
    sa = np.concatenate([states, actions], axis=-1).astype(np.float32)
    x = torch.from_numpy(sa).unsqueeze(0)  # [1, T, D]
    with torch.no_grad():
        g = model(x).squeeze(0).squeeze(-1)  # [T]
        r = torch.zeros_like(g)
        r[0] = g[0]
        r[1:] = g[1:] - g[:-1]
        r = r + (baseline / len(r))
    return r.cpu().numpy()


def spearman_corr(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def roc_auc_binary(y_true, y_score):
    y_true = y_true.astype(np.int32)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.argsort(np.argsort(y_score)) + 1
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# =========================
# Load
# =========================
ckpt = torch.load(model_path, map_location="cpu")
model = build_model_from_ckpt(ckpt)
baseline = float(ckpt.get("baseline", 0.0))

with open(data_path, "rb") as f:
    payload = pickle.load(f)
trajectories = payload["trajectories"]


# =========================
# Step-wise eval
# =========================
all_true_reward = []
all_pred_reward = []

per_traj_mae = []
per_traj_corr = []

for traj in trajectories:
    s = np.asarray(traj["states"], dtype=np.float32)
    a = np.asarray(traj["actions"], dtype=np.float32)
    r = np.asarray(traj["rewards"], dtype=np.float32).reshape(-1)

    t = min(len(s), len(a), len(r))
    if t <= 1:
        continue

    s = s[:t]
    a = a[:t]
    r = r[:t]

    pred_step = redistributed_from_model(model, s, a, baseline=baseline)

    all_true_reward.append(r)
    all_pred_reward.append(pred_step)

    per_traj_mae.append(float(np.mean(np.abs(pred_step - r))))
    if np.std(r) > 1e-8 and np.std(pred_step) > 1e-8:
        per_traj_corr.append(float(np.corrcoef(pred_step, r)[0, 1]))

all_true_reward = np.concatenate(all_true_reward)
all_pred_reward = np.concatenate(all_pred_reward)

mae = float(np.mean(np.abs(all_pred_reward - all_true_reward)))
rmse = float(np.sqrt(np.mean((all_pred_reward - all_true_reward) ** 2)))
pearson = float(np.corrcoef(all_pred_reward, all_true_reward)[0, 1])
spearman = spearman_corr(all_pred_reward, all_true_reward)

# positive-reward detection view: true reward > threshold
y_true_positive = (all_true_reward > positive_reward_threshold).astype(np.int32)
auc = roc_auc_binary(y_true_positive, all_pred_reward)

print("=== Step-wise Reward Evaluation ===")
print(f"Num trajectories used: {len(per_traj_mae)}")
print(f"Num steps used: {len(all_true_reward)}")
print(f"Global MAE: {mae:.6f}")
print(f"Global RMSE: {rmse:.6f}")
print(f"Global Pearson corr: {pearson:.4f}")
print(f"Global Spearman corr: {spearman:.4f}")
print(f"Positive-reward AUROC (true reward > {positive_reward_threshold}): {auc:.4f}")
print(f"Per-trajectory MAE mean/std: {np.mean(per_traj_mae):.6f} / {np.std(per_traj_mae):.6f}")

if len(per_traj_corr) > 0:
    print(f"Per-trajectory Pearson mean/std: {np.mean(per_traj_corr):.4f} / {np.std(per_traj_corr):.4f}")

print("\nFirst 30 step pairs (true_reward, pred_step):")
for i in range(min(30, len(all_true_reward))):
    print(f"{i:02d}: {all_true_reward[i]:.4f}, {all_pred_reward[i]:.4f}")

positive_idx = np.where(all_true_reward > positive_reward_threshold)[0]
print("\nFirst 30 pairs where true_reward > threshold (true_reward, pred_step):")
if len(positive_idx) == 0:
    print("No steps found with true_reward > threshold.")
else:
    for k, idx in enumerate(positive_idx[:30]):
        print(f"{k:02d}: {all_true_reward[idx]:.4f}, {all_pred_reward[idx]:.4f}")
