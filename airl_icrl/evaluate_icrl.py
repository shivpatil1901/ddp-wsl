"""
Evaluation script for learned reward function from AIRL ICRL checkpoint.

This script:
1. Loads the checkpoint from active_airl training
2. Extracts and reconstructs the RewardNet
3. Evaluates reward quality using metrics:
   - Spearman correlation with environment rewards
   - Pearson correlation
   - EPIC distance (canonicalized correlation)
   - Acid test: reward vs. goal distance
   - Top/Bottom transition analysis
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "airl"))


# ============================================================================
# NETWORK ARCHITECTURES (copied from airl_icrl.py for checkpoint loading)
# ============================================================================

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class RewardNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = build_mlp(state_dim * 2 + action_dim, hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions, next_states], dim=-1)
        return self.net(x).squeeze(-1)


class TaskRewardNet(nn.Module):
    """Disentangled task reward network that depends on (s, a) only."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = build_mlp(state_dim + action_dim, hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================================
# EVALUATION METRICS (adapted from reward_correlation.py)
# ============================================================================

def canonicalize_reward(rewards, states, next_states, gamma=0.99):
    """
    Approximates the canonical form of the reward to remove shaping.
    C(s,a,s') = R(s,a,s') + gamma*E[V(s')] - V(s)
    
    In a simplified offline setting, we use the mean reward as a baseline.
    """
    return rewards + gamma * np.mean(rewards) - np.mean(rewards)


def compute_correlations(env_rewards, learned_rewards):
    """
    Compute standard correlation metrics.
    """
    pearson_corr, pearson_pval = pearsonr(env_rewards, learned_rewards)
    spearman_corr, spearman_pval = spearmanr(env_rewards, learned_rewards)
    
    return {
        'pearson': pearson_corr,
        'pearson_pval': pearson_pval,
        'spearman': spearman_corr,
        'spearman_pval': spearman_pval,
    }


def compute_epic_distance(env_rewards, learned_rewards, states=None, next_states=None, gamma=0.99):
    """
    Compute EPIC-like distance as 1 - Pearson correlation between canonicalized rewards.
    Canonicalization removes shaping-like offsets before correlation.
    
    EPIC Distance ranges from 0 (perfect agreement) to 2 (perfect disagreement).
    """
    can_env = canonicalize_reward(env_rewards, states, next_states, gamma)
    can_learned = canonicalize_reward(learned_rewards, states, next_states, gamma)

    epic_corr, epic_pval = pearsonr(can_env, can_learned)
    epic_distance = 1 - epic_corr
    
    return {
        'epic_correlation': epic_corr,
        'epic_distance': epic_distance,
        'epic_pval': epic_pval,
    }


def compute_cost_correlation(costs, learned_rewards):
    """
    Check if learned reward is inversely correlated with costs (reward should penalize costs).
    """
    if costs is None or len(costs) == 0:
        return None
    
    cost_corr, cost_pval = spearmanr(costs, learned_rewards)
    return {
        'cost_correlation': cost_corr,
        'cost_pval': cost_pval,
        'cost_anti_corr': -cost_corr,  # Should be positive (inverse correlation)
    }


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_reward_net(checkpoint_path: str, state_dim: int, action_dim: int, device: str = "cpu") -> Tuple[nn.Module, bool]:
    """
    Load reward model from AIRL ICRL checkpoint.
    Returns (model, requires_next_state).
    """
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'reward_net_state' in checkpoint:
        reward_net_state = checkpoint['reward_net_state']
    elif 'airl_module' in checkpoint:
        airl_state = checkpoint['airl_module']
        if not isinstance(airl_state, dict):
            raise KeyError("airl_module is not a dict in checkpoint")

        if 'r_net_state_dict' in airl_state:
            reward_net_state = airl_state['r_net_state_dict']
        elif 'reward_net_state_dict' in airl_state:
            reward_net_state = airl_state['reward_net_state_dict']
        else:
            raise KeyError(f"Could not find reward state dict in airl_module. Keys: {airl_state.keys()}")
    else:
        raise KeyError(f"Checkpoint does not contain reward state. Keys: {checkpoint.keys()}")

    first_weight = None
    for key, tensor in reward_net_state.items():
        if key.endswith('0.weight'):
            first_weight = tensor
            break

    if first_weight is None:
        raise KeyError("Could not infer reward model architecture from state dict")

    input_dim = int(first_weight.shape[1])
    sa_dim = state_dim + action_dim
    sas_dim = state_dim * 2 + action_dim

    if input_dim == sa_dim:
        reward_net = TaskRewardNet(state_dim, action_dim).to(device)
        requires_next_state = False
    elif input_dim == sas_dim:
        reward_net = RewardNet(state_dim, action_dim).to(device)
        requires_next_state = True
    else:
        raise ValueError(
            f"Unexpected reward input dim {input_dim}. Expected {sa_dim} (s,a) or {sas_dim} (s,a,s')."
        )

    # Support both key layouts:
    # 1) "net.0.weight" style (TaskRewardNet/RewardNet wrapper)
    # 2) "0.weight" style (raw nn.Sequential saved directly)
    model_keys = set(reward_net.state_dict().keys())
    state_keys = set(reward_net_state.keys())

    if not model_keys.intersection(state_keys):
        prefixed = {f"net.{k}": v for k, v in reward_net_state.items()}
        if model_keys.intersection(prefixed.keys()):
            reward_net_state = prefixed
        else:
            unprefixed = {
                (k[4:] if k.startswith("net.") else k): v
                for k, v in reward_net_state.items()
            }
            if model_keys.intersection(unprefixed.keys()):
                reward_net_state = unprefixed

    reward_net.load_state_dict(reward_net_state)
    reward_net.eval()
    return reward_net, requires_next_state


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve checkpoint path, handling Windows UNC paths, relative paths, and repo-root-relative paths.
    """
    # Normalize path separators
    checkpoint_path = checkpoint_path.replace("\\", "/")
    
    # Handle Windows UNC path to WSL (\\wsl.localhost\Ubuntu-18.04\... -> /...)
    if checkpoint_path.startswith("wsl.localhost"):
        # Extract the path after the distro name (e.g., Ubuntu-18.04)
        parts = checkpoint_path.split("/")
        if len(parts) > 2:
            # Remove "wsl.localhost" and distro name, keep the rest with leading /
            checkpoint_path = "/" + "/".join(parts[2:])
    elif checkpoint_path.startswith("//wsl.localhost"):
        checkpoint_path = checkpoint_path.replace("//wsl.localhost/Ubuntu-18.04", "")
    
    # Try as-is first
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    
    # Try relative to current directory
    cwd_path = os.path.join(os.getcwd(), checkpoint_path)
    if os.path.isfile(cwd_path):
        return cwd_path
    
    # Try relative to repo root (parent of airl_icrl)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_path = os.path.join(repo_root, checkpoint_path)
    if os.path.isfile(repo_path):
        return repo_path
    
    # If none exist, print debug info and raise error
    print(f"ERROR: Could not find checkpoint at any of these locations:")
    print(f"  1. Normalized path: {checkpoint_path}")
    print(f"  2. CWD-relative: {cwd_path}")
    print(f"  3. Repo-root-relative: {repo_path}")
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Repo root: {repo_root}")
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def get_checkpoint_metadata(checkpoint_path: str) -> dict:
    """
    Extract metadata from checkpoint (run info, training steps, etc).
    Checkpoint structure:
        'epoch': int
        'metadata': dict
    """
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    metadata = {}
    
    if 'metadata' in checkpoint:
        metadata.update(checkpoint['metadata'])
    
    if 'epoch' in checkpoint:
        metadata['epoch'] = int(checkpoint['epoch'])
    
    # Extract lambda from generator state if available
    if 'generator' in checkpoint and 'log_lambda' in checkpoint['generator']:
        lambda_val = float(checkpoint['generator']['log_lambda'].exp().item()) if hasattr(checkpoint['generator']['log_lambda'], 'exp') else float(checkpoint['generator']['log_lambda'])
        metadata['lambda'] = lambda_val
    
    return metadata


# ============================================================================
# EXPERT DATA LOADING
# ============================================================================

def load_expert_data(expert_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load expert trajectory data from pickle file.
    Returns: states, actions, next_states, rewards, costs (or None if not available)
    """
    # Resolve path using same logic as checkpoint
    original_path = expert_path
    expert_path = expert_path.replace("\\", "/")
    
    # Try as-is first
    if os.path.isfile(expert_path):
        pass  # Use expert_path as-is
    # Try relative to current directory
    elif os.path.isfile(os.path.join(os.getcwd(), expert_path)):
        expert_path = os.path.join(os.getcwd(), expert_path)
    # Try relative to repo root (parent of airl_icrl)
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_path = os.path.join(repo_root, expert_path)
        if os.path.isfile(repo_path):
            expert_path = repo_path
        else:
            print(f"ERROR: Could not find expert data at any of these locations:")
            print(f"  1. As-is: {original_path}")
            print(f"  2. CWD-relative: {os.path.join(os.getcwd(), original_path)}")
            print(f"  3. Repo-root-relative: {repo_path}")
            raise FileNotFoundError(f"Expert data not found: {original_path}")
    
    print(f"Loading expert data from {expert_path}...")
    
    with open(expert_path, 'rb') as f:
        try:
            import pickle5 as pkl
            expert_demo = pkl.load(f)
        except ImportError:
            expert_demo = pickle.load(f)
    
    if not isinstance(expert_demo, dict):
        raise TypeError(
            f"Expected expert demo to be a dict, got {type(expert_demo)}"
        )

    # Some datasets store trajectories under a nested payload.
    payload = expert_demo
    for nested_key in ("data", "dataset", "trajectories"):
        if nested_key in payload and isinstance(payload[nested_key], dict):
            payload = payload[nested_key]
            break

    def _pick(mapping: dict, candidates, field_name: str, required: bool = True):
        for key in candidates:
            if key in mapping:
                return mapping[key]
        if required:
            raise KeyError(
                f"Missing '{field_name}' in expert data. Tried keys={list(candidates)}. "
                f"Available keys={list(mapping.keys())}"
            )
        return None

    states_raw = _pick(payload, ("states", "observations", "obs", "state", "s"), "states")
    actions_raw = _pick(payload, ("actions", "acts", "action", "a"), "actions")
    next_states_raw = _pick(
        payload,
        ("next_states", "next_observations", "next_obs", "next_state", "s_next", "obs2"),
        "next_states",
    )
    rewards_raw = _pick(payload, ("rewards", "reward", "rews", "r", "env_rewards"), "rewards")
    costs_raw = _pick(payload, ("costs", "cost", "c"), "costs", required=False)

    states = np.array(states_raw, dtype=np.float32)
    actions = np.array(actions_raw, dtype=np.float32)
    next_states = np.array(next_states_raw, dtype=np.float32)
    rewards = np.array(rewards_raw, dtype=np.float32).flatten()

    costs = None
    if costs_raw is not None:
        costs = np.array(costs_raw, dtype=np.float32).flatten()

    n = min(len(states), len(actions), len(next_states), len(rewards))
    if costs is not None:
        n = min(n, len(costs))
    if n == 0:
        raise ValueError("Expert data is empty after loading")
    if not (len(states) == len(actions) == len(next_states) == len(rewards) and (costs is None or len(costs) == len(states))):
        print(
            "WARNING: Mismatched trajectory lengths "
            f"(states={len(states)}, actions={len(actions)}, next_states={len(next_states)}, "
            f"rewards={len(rewards)}, costs={len(costs) if costs is not None else 'N/A'}). "
            f"Truncating all to {n}."
        )
        states = states[:n]
        actions = actions[:n]
        next_states = next_states[:n]
        rewards = rewards[:n]
        if costs is not None:
            costs = costs[:n]
    
    print(f"Loaded {len(states)} transitions")
    print(f"  State shape: {states.shape}, Action shape: {actions.shape}")
    print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}], mean: {rewards.mean():.3f}")
    if costs is not None:
        print(f"  Cost range: [{costs.min():.3f}, {costs.max():.3f}], mean: {costs.mean():.3f}")
    
    return states, actions, next_states, rewards, costs


# ============================================================================
# REWARD COMPUTATION
# ============================================================================

def compute_learned_rewards(reward_net: nn.Module, states: np.ndarray, actions: np.ndarray,
                           next_states: np.ndarray, batch_size: int = 5000, device: str = "cpu",
                           requires_next_state: bool = True) -> np.ndarray:
    """
    Compute learned rewards using the loaded reward model (batch processing).
    """
    print("Computing learned rewards...")
    learned_rewards = []

    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            s_batch = torch.from_numpy(states[i:i+batch_size]).to(device)
            a_batch = torch.from_numpy(actions[i:i+batch_size]).to(device)
            ns_batch = torch.from_numpy(next_states[i:i+batch_size]).to(device)

            if requires_next_state:
                r_pred = reward_net(s_batch, a_batch, ns_batch)
            else:
                r_pred = reward_net(s_batch, a_batch)
            learned_rewards.extend(r_pred.cpu().numpy().flatten())

    return np.array(learned_rewards)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_reward_comparison(env_rewards: np.ndarray, learned_rewards: np.ndarray, 
                          output_dir: str, sample_size: int = 5000):
    """
    Create scatter plot of environment vs learned rewards with trend line.
    """
    # Sample for cleaner visualization
    if len(env_rewards) > sample_size:
        idx = np.random.choice(len(env_rewards), sample_size, replace=False)
    else:
        idx = np.arange(len(env_rewards))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(env_rewards[idx], learned_rewards[idx], alpha=0.3, s=10, color='blue')
    
    # Add trend line
    z = np.polyfit(env_rewards[idx], learned_rewards[idx], 1)
    p = np.poly1d(z)
    plt.plot(env_rewards[idx], p(env_rewards[idx]), "r--", linewidth=2, label="Trend Line")
    
    plt.xlabel("Environment Ground Truth Reward", fontsize=12)
    plt.ylabel("Learned Reward g(s, a, s')", fontsize=12)
    plt.title("Learned Reward vs. Ground Truth Reward", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, "reward_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_goal_distance_vs_reward(states: np.ndarray, learned_rewards: np.ndarray, 
                                  output_dir: str, sample_size: int = 5000):
    """
    Acid test: learned reward vs goal distance (assumes states[:, :2] is goal position).
    """
    # Goal distance from origin
    goal_dist = np.linalg.norm(states[:, :2], axis=1)
    
    # Sample
    if len(goal_dist) > sample_size:
        idx = np.random.choice(len(goal_dist), sample_size, replace=False)
    else:
        idx = np.arange(len(goal_dist))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(goal_dist[idx], learned_rewards[idx], alpha=0.4, s=10, color='purple')
    
    # Add trend line
    z = np.polyfit(goal_dist[idx], learned_rewards[idx], 1)
    p = np.poly1d(z)
    plt.plot(goal_dist[idx], p(goal_dist[idx]), "r--", linewidth=2, label="Trend Line")
    
    plt.xlabel("Relative Distance to Goal (Lower = Closer)", fontsize=12)
    plt.ylabel("Learned Reward g(s, a, s')", fontsize=12)
    plt.title("Acid Test: Learned Reward vs. Goal Distance\n(Reward should increase as goal gets closer)", fontsize=14)
    plt.gca().invert_xaxis()  # Invert so moving forward is left to right
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, "goal_distance_vs_reward.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_cost_vs_reward(costs: np.ndarray, learned_rewards: np.ndarray, 
                       output_dir: str, sample_size: int = 5000):
    """
    Plot learned reward vs costs (should be negatively correlated).
    """
    # Sample
    if len(costs) > sample_size:
        idx = np.random.choice(len(costs), sample_size, replace=False)
    else:
        idx = np.arange(len(costs))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(costs[idx], learned_rewards[idx], alpha=0.3, s=10, color='orange')
    
    # Add trend line
    z = np.polyfit(costs[idx], learned_rewards[idx], 1)
    p = np.poly1d(z)
    plt.plot(costs[idx], p(costs[idx]), "r--", linewidth=2, label="Trend Line")
    
    plt.xlabel("Cost (Hazard Violations)", fontsize=12)
    plt.ylabel("Learned Reward g(s, a, s')", fontsize=12)
    plt.title("Safety Validation: Learned Reward vs. Cost\n(Should be negatively correlated)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, "cost_vs_reward.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

def analyze_top_bottom_transitions(env_rewards: np.ndarray, learned_rewards: np.ndarray, 
                                   costs: Optional[np.ndarray], num_transitions: int = 5):
    """
    Analyze the top and bottom transitions according to learned reward.
    """
    sorted_idx = np.argsort(learned_rewards)[::-1]
    
    print("\n" + "="*80)
    print(f"TOP {num_transitions} EXPERT TRANSITIONS (By Learned Reward)")
    print("="*80)
    print(f"{'Rank':<6} {'Learned':<12} {'Ground Truth':<15} {'Cost':<8} {'Match':<8}")
    print("-"*80)
    
    for i in range(num_transitions):
        idx = sorted_idx[i]
        cost_str = f"{costs[idx]:.3f}" if costs is not None else "N/A"
        # Check if learned ranking matches ground truth (within top 10%)
        gt_rank = int(np.argsort(env_rewards)[::-1].tolist().index(idx))
        is_match = "✓" if gt_rank < int(0.1 * len(env_rewards)) else "✗"
        
        print(f"{i+1:<6} {learned_rewards[idx]:<12.3f} {env_rewards[idx]:<15.3f} {cost_str:<8} {is_match:<8}")
    
    print("\n" + "="*80)
    print(f"BOTTOM {num_transitions} EXPERT TRANSITIONS (By Learned Reward)")
    print("="*80)
    print(f"{'Rank':<6} {'Learned':<12} {'Ground Truth':<15} {'Cost':<8} {'Match':<8}")
    print("-"*80)
    
    for i in range(1, num_transitions + 1):
        idx = sorted_idx[-i]
        cost_str = f"{costs[idx]:.3f}" if costs is not None else "N/A"
        gt_rank = int(np.argsort(env_rewards).tolist().index(idx))
        is_match = "✓" if gt_rank < int(0.1 * len(env_rewards)) else "✗"
        
        print(f"{len(learned_rewards)-i+1:<6} {learned_rewards[idx]:<12.3f} {env_rewards[idx]:<15.3f} {cost_str:<8} {is_match:<8}")


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_learned_reward(checkpoint_path: str, expert_path: str, output_dir: str = "./eval_results", 
                           state_dim: int = 60, action_dim: int = 2, device: str = "cpu"):
    """
    Full evaluation pipeline for learned reward function.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("EVALUATING LEARNED REWARD FUNCTION")
    print("="*80)
    
    # Load checkpoint metadata
    print("\n[1/5] Loading checkpoint metadata...")
    metadata = get_checkpoint_metadata(checkpoint_path)
    print(f"  Checkpoint metadata: {metadata}")
    
    # Load reward net
    print("\n[2/5] Loading reward network from checkpoint...")
    reward_net, requires_next_state = load_reward_net(checkpoint_path, state_dim, action_dim, device=device)
    mode = "(s,a,s')" if requires_next_state else "(s,a)"
    print(f"  Reward network loaded and set to eval mode, input mode: {mode}")
    
    # Load expert data
    print("\n[3/5] Loading expert data...")
    states, actions, next_states, env_rewards, costs = load_expert_data(expert_path)
    
    # Compute learned rewards
    print("\n[4/5] Computing learned rewards...")
    learned_rewards = compute_learned_rewards(
        reward_net,
        states,
        actions,
        next_states,
        device=device,
        requires_next_state=requires_next_state,
    )
    print(f"  Learned reward range: [{learned_rewards.min():.3f}, {learned_rewards.max():.3f}]")
    print(f"  Mean learned reward: {learned_rewards.mean():.3f}, Std: {learned_rewards.std():.3f}")
    
    # Compute evaluation metrics
    print("\n[5/5] Computing evaluation metrics...")
    
    # Basic correlations
    corr_metrics = compute_correlations(env_rewards, learned_rewards)
    
    # EPIC distance (canonicalized rewards)
    epic_metrics = compute_epic_distance(env_rewards, learned_rewards, states, next_states)
    
    # Cost correlation
    cost_metrics = compute_cost_correlation(costs, learned_rewards)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\n[Correlation Metrics]")
    print(f"  Pearson Correlation:  {corr_metrics['pearson']:>8.4f} (p-value: {corr_metrics['pearson_pval']:.2e})")
    print(f"  Spearman Correlation: {corr_metrics['spearman']:>8.4f} (p-value: {corr_metrics['spearman_pval']:.2e})")
    
    print("\n[EPIC Metrics (Canonicalized)]")
    print(f"  EPIC Correlation: {epic_metrics['epic_correlation']:>8.4f}")
    print(f"  EPIC Distance:    {epic_metrics['epic_distance']:>8.4f} (lower is better, 0=perfect)")
    
    if cost_metrics is not None:
        print("\n[Safety Metrics]")
        print(f"  Cost Correlation:      {cost_metrics['cost_correlation']:>8.4f}")
        print(f"  Cost Anti-Correlation: {cost_metrics['cost_anti_corr']:>8.4f} (higher is better)")
        print(f"  (Should be negative: reward should penalize costs)")
    
    # Detailed analysis
    print("\n" + "="*80)
    analyze_top_bottom_transitions(env_rewards, learned_rewards, costs, num_transitions=5)
    print("="*80)
    
    # Generate plots
    print("\n[Generating Visualizations]")
    plot_reward_comparison(env_rewards, learned_rewards, output_dir)
    plot_goal_distance_vs_reward(states, learned_rewards, output_dir)
    if costs is not None:
        plot_cost_vs_reward(costs, learned_rewards, output_dir)
    
    # Save summary report
    print("\n[Saving Summary Report]")
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("LEARNED REWARD EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Expert Data: {expert_path}\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("[Checkpoint Metadata]\n")
        for key, val in metadata.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")
        
        f.write("[Expert Data]\n")
        f.write(f"  Total transitions: {len(states)}\n")
        f.write(f"  State dim: {states.shape[1]}, Action dim: {actions.shape[1]}\n")
        f.write(f"  Env reward range: [{env_rewards.min():.3f}, {env_rewards.max():.3f}]\n")
        f.write(f"  Learned reward range: [{learned_rewards.min():.3f}, {learned_rewards.max():.3f}]\n\n")
        
        f.write("[Correlation Metrics]\n")
        f.write(f"  Pearson: {corr_metrics['pearson']:.4f}\n")
        f.write(f"  Spearman: {corr_metrics['spearman']:.4f}\n\n")
        
        f.write("[EPIC Metrics]\\n")
        f.write(f"  EPIC Correlation: {epic_metrics['epic_correlation']:.4f}\\n")
        f.write(f"  EPIC Distance: {epic_metrics['epic_distance']:.4f}\\n\\n")
        
        if cost_metrics is not None:
            f.write("[Safety Metrics]\n")
            f.write(f"  Cost Correlation: {cost_metrics['cost_correlation']:.4f}\n")
            f.write(f"  Cost Anti-Correlation: {cost_metrics['cost_anti_corr']:.4f}\n\n")
        
        f.write("[Interpretation]\n")
        if epic_metrics['epic_distance'] < 0.3:
            f.write("  ✓ GOOD: EPIC distance < 0.3 suggests reward function is well-learned\\n")
        elif epic_metrics['epic_distance'] < 0.5:
            f.write("  ~ FAIR: EPIC distance 0.3-0.5 suggests moderate learning quality\\n")
        else:
            f.write("  ✗ POOR: EPIC distance > 0.5 suggests poor reward learning\\n")
        
        if corr_metrics['spearman'] > 0.5:
            f.write("  ✓ GOOD: Spearman > 0.5 suggests rank ordering is preserved\n")
        elif corr_metrics['spearman'] > 0.0:
            f.write("  ~ FAIR: Spearman near 0 suggests weak ranking correlation\n")
        else:
            f.write("  ✗ POOR: Negative Spearman suggests inverse ranking\n")
        
        if cost_metrics and cost_metrics['cost_anti_corr'] > 0.3:
            f.write("  ✓ GOOD: Anti-correlation > 0.3 suggests safety awareness\n")
        elif cost_metrics:
            f.write("  ✗ POOR: Weak anti-correlation with costs; reward not safety-aware\n")
    
    print(f"Report saved to {report_path}")
    
    print("\n" + "="*80)
    print(f"Evaluation complete! Results saved to {output_dir}")
    print("="*80 + "\n")
    
    return {
        'metadata': metadata,
        'correlation_metrics': corr_metrics,
        'epic_metrics': epic_metrics,
        'cost_metrics': cost_metrics,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import time
    
    parser = argparse.ArgumentParser(description="Evaluate learned reward function from AIRL ICRL checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to checkpoint_final.pt from active AIRL training")
    parser.add_argument("--expert_data", type=str, required=True,
                       help="Path to expert data pickle file (e.g., ppo_lagrangian_PointGoal1_s0.pickle)")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Directory to save evaluation results (default: ./eval_results)")
    parser.add_argument("--state_dim", type=int, default=60,
                       help="State dimension (default: 60 for PointGoal1)")
    parser.add_argument("--action_dim", type=int, default=2,
                       help="Action dimension (default: 2 for PointGoal1)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for computation (cpu/cuda, default: cpu)")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_learned_reward(
        checkpoint_path=args.checkpoint,
        expert_path=args.expert_data,
        output_dir=args.output_dir,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        device=args.device,
    )
