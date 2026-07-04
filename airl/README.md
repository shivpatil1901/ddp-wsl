# AIRL with Frozen SafeDICE Generator

This implementation trains an AIRL (Adversarial Inverse Reinforcement Learning) discriminator to learn the reward function using a frozen SafeDICE policy as the generator.

## Overview

**AIRL** learns a reward function from expert demonstrations by training a discriminator to distinguish between expert and policy trajectories. The key advantage of AIRL over GAIL is that it learns a **disentangled reward function** that is invariant to changes in dynamics.

### Key Components

1. **Frozen Generator**: Pre-trained SafeDICE policy (not updated)
2. **AIRL Discriminator**: Learns reward function with structure:
   - `f(s, a, s') = g(s, a) + γh(s') - h(s)`
   - Where `g(s,a)` is reward shaping and `h(s)` is value shaping

## Usage

### 1. Train AIRL Discriminator

```bash
cd airl

# Activate environment
conda activate safedice

# Train AIRL discriminator with frozen SafeDICE policy
python airl_safedice.py \
    --checkpoint_path ../SafeDICE/weights/antidice_PointGoal1_seed0_20260206_184622_iter1000000.pickle \
    --robot_name Point \
    --task_name Goal1 \
    --num_expert_trajectories 1000 \
    --num_policy_rollouts 100 \
    --num_iterations 10000 \
    --batch_size 256 \
    --learning_rate 3e-4 \
    --gamma 0.99 \
    --seed 0 \
    --log_interval 100 \
    --rollout_interval 1000
```

**Arguments:**
- `--checkpoint_path`: Path to trained SafeDICE checkpoint (required)
- `--num_expert_trajectories`: Number of expert trajectories to use (default: 1000)
- `--num_policy_rollouts`: Policy rollouts per collection (default: 100)
- `--num_iterations`: Total training iterations (default: 10000)
- `--batch_size`: Batch size for discriminator training (default: 256)
- `--rollout_interval`: Collect new rollouts every N iterations (default: 1000)

**Output:**
- Logs saved to: `./airl_logs/airl_PointGoal1_seed0_<timestamp>.csv`
- Checkpoints saved to: `./airl_weights/airl_disc_PointGoal1_iter<N>.pkl`

### 2. Evaluate Learned Reward Function

```bash
python evaluate_airl.py \
    --safedice_checkpoint ../SafeDICE/weights/antidice_PointGoal1_seed0_20260206_184622_iter1000000.pickle \
    --airl_checkpoint ./airl_weights/airl_disc_PointGoal1_final.pkl \
    --robot_name Point \
    --task_name Goal1 \
    --num_expert_trajectories 100 \
    --num_policy_rollouts 50 \
    --visualize
```

**Output:**
- Classification accuracies (expert vs policy)
- Reward statistics comparison
- Visualization plots (if `--visualize` flag is used)

## Architecture Details

### AIRL Discriminator

```python
f(s, a, s') = g(s, a) + γh(s') - h(s)

D(s, a, s') = exp(f(s, a, s')) / (exp(f(s, a, s')) + π(a|s))
            = sigmoid(f(s, a, s') - log π(a|s))

Learned Reward: r(s, a, s') = f(s, a, s') - log π(a|s)
```

**Networks:**
- `g(s, a)`: 2-layer MLP (256 hidden units) → scalar
- `h(s)`: 2-layer MLP (256 hidden units) → scalar

### Training Objective

```
min_D  -E_expert[log D(s,a,s')] - E_policy[log(1 - D(s,a,s'))]
       + λ * gradient_penalty
```

**Gradient Penalty** ensures discriminator stability (λ=10.0)

## Implementation Features

✅ **Frozen Generator**: SafeDICE policy is never updated
✅ **Reward Disentanglement**: AIRL's structure ensures learned reward is dynamics-invariant
✅ **Periodic Rollouts**: Collects fresh policy rollouts to keep buffer diverse
✅ **Gradient Penalty**: Lipschitz constraint for stable training
✅ **CSV Logging**: All metrics logged for analysis
✅ **Checkpointing**: Regular model saves for evaluation

## Expected Results

Good discriminator performance:
- Expert accuracy: > 0.8 (classifies expert as expert)
- Policy accuracy: > 0.8 (classifies policy as policy)
- Expert D(s,a,s'): > 0.7
- Policy D(s,a,s'): < 0.3

The learned reward should assign higher values to expert trajectories compared to policy trajectories.

## File Structure

```
airl/
├── airl_safedice.py       # Main training script
├── evaluate_airl.py       # Evaluation script
├── README.md              # This file
├── airl_logs/            # Training logs (CSV)
└── airl_weights/         # Saved discriminator models
```

## Reference

Based on:
- **AIRL Paper**: "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning" (Fu et al., 2017)
- **Implementation**: https://github.com/toshikwa/gail-airl-ppo.pytorch

## Troubleshooting

**Issue**: Out of memory during rollouts
- Solution: Reduce `--num_policy_rollouts` or run on CPU

**Issue**: Discriminator not learning (acc ~0.5)
- Solution: Increase `--num_iterations`, try different `--learning_rate`

**Issue**: Expert/policy rewards very similar
- Solution: Check if SafeDICE policy is too similar to expert, may need more diverse policy

## Next Steps

After training the AIRL discriminator:
1. Use learned reward `r(s,a,s')` for policy optimization
2. Compare with ground-truth reward from environment
3. Transfer learned reward to different dynamics
