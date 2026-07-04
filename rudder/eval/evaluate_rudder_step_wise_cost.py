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
model_path = os.path.join(repo_root, "rudder", "cost_rudder.pt")
data_path = os.path.join(repo_root, "rudder", "combined_cost_balanced_1400.pkl")
hazard_threshold = 0.0  # cost > 0 treated as hazardous step


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


def redistributed_from_model(model, states, actions, baseline=0.0):
    sa = np.concatenate([states, actions], axis=-1).astype(np.float32)
    x = torch.from_numpy(sa).unsqueeze(0)  # [1, T, D]
    with torch.no_grad():
        g = model(x).squeeze(0).squeeze(-1)  # [T], cumulative-like signal
        r = torch.zeros_like(g)
        r[0] = g[0]
        r[1:] = g[1:] - g[:-1]
        # same baseline redistribution as training code
        r = r + (baseline / len(r))
    return r.cpu().numpy()


def spearman_corr(x, y):
    # no scipy dependency
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def roc_auc_binary(y_true, y_score):
    # Mann-Whitney U / rank-based AUC, no sklearn dependency
    y_true = y_true.astype(np.int32)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.argsort(np.argsort(y_score)) + 1  # 1..N
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# =========================
# Load
# =========================
ckpt = torch.load(model_path, map_location="cpu")
hidden_dim = ckpt["model_state_dict"]["lstm.weight_ih_l0"].shape[0] // 4
model = OriginalRUDDER(
    state_dim=int(ckpt["state_dim"]),
    action_dim=int(ckpt["action_dim"]),
    hidden_dim=hidden_dim,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
baseline = float(ckpt.get("baseline", 0.0))

with open(data_path, "rb") as f:
    payload = pickle.load(f)
trajectories = payload["trajectories"]

# =========================
# Step-wise eval
# =========================
all_true_cost = []
all_pred_cost = []

per_traj_mae = []
per_traj_corr = []

for traj in trajectories:
    s = np.asarray(traj["states"], dtype=np.float32)
    a = np.asarray(traj["actions"], dtype=np.float32)
    c = np.asarray(traj["costs"], dtype=np.float32).reshape(-1)

    T = min(len(s), len(a), len(c))
    if T <= 1:
        continue

    s = s[:T]
    a = a[:T]
    c = c[:T]

    pred_step = redistributed_from_model(model, s, a, baseline=baseline)

    all_true_cost.append(c)
    all_pred_cost.append(pred_step)

    per_traj_mae.append(float(np.mean(np.abs(pred_step - c))))
    if np.std(c) > 1e-8 and np.std(pred_step) > 1e-8:
        per_traj_corr.append(float(np.corrcoef(pred_step, c)[0, 1]))

all_true_cost = np.concatenate(all_true_cost)
all_pred_cost = np.concatenate(all_pred_cost)

mae = float(np.mean(np.abs(all_pred_cost - all_true_cost)))
rmse = float(np.sqrt(np.mean((all_pred_cost - all_true_cost) ** 2)))
pearson = float(np.corrcoef(all_pred_cost, all_true_cost)[0, 1])
spearman = spearman_corr(all_pred_cost, all_true_cost)

# hazard detection view: true cost > 0
y_true_hazard = (all_true_cost > hazard_threshold).astype(np.int32)
auc = roc_auc_binary(y_true_hazard, all_pred_cost)

print("=== Step-wise Cost Evaluation ===")
print(f"Num trajectories used: {len(per_traj_mae)}")
print(f"Num steps used: {len(all_true_cost)}")
print(f"Global MAE: {mae:.6f}")
print(f"Global RMSE: {rmse:.6f}")
print(f"Global Pearson corr: {pearson:.4f}")
print(f"Global Spearman corr: {spearman:.4f}")
print(f"Hazard AUROC (true cost > {hazard_threshold}): {auc:.4f}")
print(f"Per-trajectory MAE mean/std: {np.mean(per_traj_mae):.6f} / {np.std(per_traj_mae):.6f}")

if len(per_traj_corr) > 0:
    print(f"Per-trajectory Pearson mean/std: {np.mean(per_traj_corr):.4f} / {np.std(per_traj_corr):.4f}")

# quick peek
print("\nFirst 30 step pairs (true_cost, pred_step):")
for i in range(min(30, len(all_true_cost))):
    print(f"{i:02d}: {all_true_cost[i]:.4f}, {all_pred_cost[i]:.4f}")

# inspect predictions specifically on hazardous steps (true cost == 1)
hazard_idx = np.where(all_true_cost == 1.0)[0]
print("\nFirst 30 pairs where true_cost == 1 (true_cost, pred_step):")
if len(hazard_idx) == 0:
    print("No steps found with true_cost == 1.")
else:
    for k, idx in enumerate(hazard_idx[:30]):
        print(f"{k:02d}: {all_true_cost[idx]:.4f}, {all_pred_cost[idx]:.4f}")