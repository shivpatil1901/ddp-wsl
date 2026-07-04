"""
Behavior Cloning for AIRL initialization (SafeDICE actor only).

This script trains only the actor used by SafeDICE/AIRL initialization,
logs train/val supervised losses, and evaluates policy reward on two
fixed evaluation seed sets ("train" seeds and "val" seeds).
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

import gym
import numpy as np
import safety_gym
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'SafeDICE'))

from algorithms.safedice import SafeDICE as AntiDICE


def safe_reset(env, seed=None):
    if seed is not None:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, 'seed'):
                env.seed(seed)
            out = env.reset()
    else:
        out = env.reset()

    if isinstance(out, tuple):
        return out[0]
    return out


def safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = terminated or truncated
        return next_state, reward, done, info
    return out


def load_expert_data(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except ValueError as e:
        if 'unsupported pickle protocol' not in str(e):
            raise
        import pickle5 as pickle5lib
        with open(path, 'rb') as f:
            data = pickle5lib.load(f)

    for key in ['states', 'actions', 'next_states']:
        if key not in data:
            raise ValueError('Missing key in expert data: %s' % key)

    states = np.asarray(data['states'], dtype=np.float32)
    actions = np.asarray(data['actions'], dtype=np.float32)
    next_states = np.asarray(data['next_states'], dtype=np.float32)
    n = min(len(states), len(actions), len(next_states))

    return {
        'states': states[:n],
        'actions': actions[:n],
        'next_states': next_states[:n],
    }


def split_train_val(states, actions, train_split, seed):
    n = len(states)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_split)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return (
        states[train_idx], actions[train_idx],
        states[val_idx], actions[val_idx],
    )


def make_dataset(obs, acts, batch_size, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((obs, acts))
    if shuffle:
        ds = ds.shuffle(min(10000, len(obs)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_policy(obs_dim, act_dim, actor_lr):
    config = {
        'hidden_size': 256,
        'critic_lr': actor_lr,
        'actor_lr': actor_lr,
        'grad_reg_coeffs': [10.0, 10.0],
        'gamma': 0.99,
        'alpha': 0,
        'use_last_layer_bias_cost': True,
        'use_last_layer_bias_critic': True,
        'kernel_initializer': 'glorot_uniform',
    }
    return AntiDICE(obs_dim, act_dim, mixture_actor=False, is_discrete_action=False, config=config)


@tf.function
def train_step(policy, obs, acts):
    with tf.GradientTape() as tape:
        log_probs = policy.actor.get_log_prob(obs, acts)
        nll = -tf.reduce_mean(log_probs)

    grads = tape.gradient(nll, policy.actor.variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    policy.actor_optimizer.apply_gradients(zip(grads, policy.actor.variables))
    return nll


@tf.function
def eval_batch(policy, obs, acts):
    log_probs = policy.actor.get_log_prob(obs, acts)
    nll = -tf.reduce_mean(log_probs)

    pred = policy.step(obs, deterministic=True)
    mse = tf.reduce_mean(tf.square(pred - acts))
    mae = tf.reduce_mean(tf.abs(pred - acts))
    return nll, mse, mae


def evaluate_supervised(policy, dataset):
    nlls, mses, maes = [], [], []
    for obs, acts in dataset:
        nll, mse, mae = eval_batch(policy, obs, acts)
        nlls.append(float(nll.numpy()))
        mses.append(float(mse.numpy()))
        maes.append(float(mae.numpy()))

    return {
        'nll': float(np.mean(nlls)) if nlls else 0.0,
        'mse': float(np.mean(mses)) if mses else 0.0,
        'mae': float(np.mean(maes)) if maes else 0.0,
    }


def rollout_episode(env, policy, seed, max_ep_len):
    obs = safe_reset(env, seed=seed)
    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0

    done = False
    while not done and ep_len < max_ep_len:
        obs_t = tf.convert_to_tensor([obs], dtype=tf.float32)
        act = policy.step(obs_t, deterministic=True).numpy()[0]
        act = np.clip(act, env.action_space.low, env.action_space.high)

        obs, rew, done, info = safe_step(env, act)
        ep_ret += float(rew)
        ep_cost += float(info.get('cost', 0.0))
        ep_len += 1

    return ep_ret, ep_cost, ep_len


def evaluate_reward(policy, env_name, seeds, max_ep_len):
    env = gym.make(env_name)
    rets, costs, lens = [], [], []

    for s in seeds:
        r, c, l = rollout_episode(env, policy, seed=int(s), max_ep_len=max_ep_len)
        rets.append(r)
        costs.append(c)
        lens.append(l)

    env.close()
    return {
        'reward_mean': float(np.mean(rets)) if rets else 0.0,
        'reward_std': float(np.std(rets)) if rets else 0.0,
        'cost_mean': float(np.mean(costs)) if costs else 0.0,
        'len_mean': float(np.mean(lens)) if lens else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--expert_data_path',
        type=str,
        default=os.path.join(ROOT_DIR, 'SafeDICE', 'dataset', 'safetygym', 'ppo_lagrangian_PointGoal1_s0.pickle'),
    )
    parser.add_argument('--robot_name', type=str, default='Point')
    parser.add_argument('--task_name', type=str, default='Goal1')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_split', type=float, default=0.8)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--actor_lr', type=float, default=3e-4)

    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--reward_eval_every', type=int, default=5)
    parser.add_argument('--train_reward_episodes', type=int, default=5)
    parser.add_argument('--val_reward_episodes', type=int, default=5)
    parser.add_argument('--max_ep_len', type=int, default=1000)

    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--policy_output_path', type=str, default='')

    return parser.parse_args()


def main():
    args = parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print('TensorFlow version: %s' % tf.__version__)
    print('GPU available: %s' % tf.config.list_physical_devices('GPU'))

    print('Loading expert data from: %s' % args.expert_data_path)
    expert = load_expert_data(args.expert_data_path)

    states = expert['states']
    actions = expert['actions']

    train_obs, train_acts, val_obs, val_acts = split_train_val(
        states, actions, args.train_split, args.seed
    )

    obs_dim = int(train_obs.shape[1])
    act_dim = int(train_acts.shape[1])
    env_name = 'Safexp-%s%s-v0' % (args.robot_name, args.task_name)

    print('Observation dim: %d | Action dim: %d' % (obs_dim, act_dim))
    print('Train samples: %d | Val samples: %d' % (len(train_obs), len(val_obs)))

    train_ds = make_dataset(train_obs, train_acts, args.batch_size, shuffle=True)
    train_eval_ds = make_dataset(train_obs, train_acts, args.batch_size, shuffle=False)
    val_ds = make_dataset(val_obs, val_acts, args.batch_size, shuffle=False)

    policy = build_policy(obs_dim, act_dim, args.actor_lr)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            ROOT_DIR,
            'airl',
            'airl_results',
            'bc_airl_init_%s_%s%s_s%d' % (timestamp, args.robot_name, args.task_name, args.seed),
        )
    os.makedirs(output_dir, exist_ok=True)

    if args.policy_output_path:
        policy_path = args.policy_output_path
    else:
        policy_path = os.path.join(output_dir, 'policy_final.pickle')

    metrics_path = os.path.join(output_dir, 'metrics.csv')
    config_path = os.path.join(output_dir, 'config.json')

    train_reward_seeds = [args.seed + i for i in range(args.train_reward_episodes)]
    val_reward_seeds = [args.seed + 10000 + i for i in range(args.val_reward_episodes)]

    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_nll', 'train_mse', 'train_mae',
            'val_nll', 'val_mse', 'val_mae',
            'train_reward_mean', 'train_reward_std', 'train_cost_mean',
            'val_reward_mean', 'val_reward_std', 'val_cost_mean',
        ])

        best_val_nll = float('inf')
        best_actor_weights = None

        for epoch in range(1, args.epochs + 1):
            batch_nll = []
            for obs, acts in train_ds:
                nll = train_step(policy, obs, acts)
                batch_nll.append(float(nll.numpy()))

            train_metrics = evaluate_supervised(policy, train_eval_ds)
            val_metrics = evaluate_supervised(policy, val_ds)

            if val_metrics['nll'] < best_val_nll:
                best_val_nll = val_metrics['nll']
                best_actor_weights = [v.numpy().copy() for v in policy.actor.variables]

            should_eval_reward = (epoch == 1) or (epoch % args.reward_eval_every == 0) or (epoch == args.epochs)
            if should_eval_reward:
                train_reward = evaluate_reward(policy, env_name, train_reward_seeds, args.max_ep_len)
                val_reward = evaluate_reward(policy, env_name, val_reward_seeds, args.max_ep_len)
            else:
                train_reward = {'reward_mean': np.nan, 'reward_std': np.nan, 'cost_mean': np.nan}
                val_reward = {'reward_mean': np.nan, 'reward_std': np.nan, 'cost_mean': np.nan}

            writer.writerow([
                epoch,
                train_metrics['nll'], train_metrics['mse'], train_metrics['mae'],
                val_metrics['nll'], val_metrics['mse'], val_metrics['mae'],
                train_reward['reward_mean'], train_reward['reward_std'], train_reward['cost_mean'],
                val_reward['reward_mean'], val_reward['reward_std'], val_reward['cost_mean'],
            ])
            f.flush()

            if epoch % args.log_every == 0 or epoch == 1:
                print(
                    'Epoch %d/%d | Train NLL %.4f MSE %.6f | Val NLL %.4f MSE %.6f'
                    % (epoch, args.epochs, train_metrics['nll'], train_metrics['mse'], val_metrics['nll'], val_metrics['mse'])
                )
                if should_eval_reward:
                    print(
                        '  Reward Eval | TrainSeeds %.2f+-%.2f (cost %.2f) | ValSeeds %.2f+-%.2f (cost %.2f)'
                        % (
                            train_reward['reward_mean'], train_reward['reward_std'], train_reward['cost_mean'],
                            val_reward['reward_mean'], val_reward['reward_std'], val_reward['cost_mean'],
                        )
                    )

    if best_actor_weights is not None:
        for var, val in zip(policy.actor.variables, best_actor_weights):
            var.assign(val)

    final_train_reward = evaluate_reward(policy, env_name, train_reward_seeds, args.max_ep_len)
    final_val_reward = evaluate_reward(policy, env_name, val_reward_seeds, args.max_ep_len)

    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    policy.save(
        policy_path,
        {
            'algo': 'BC_pretrain_for_AIRL',
            'seed': int(args.seed),
            'epochs': int(args.epochs),
            'expert_data_path': args.expert_data_path,
            'best_val_nll': float(best_val_nll),
            'final_train_reward_mean': final_train_reward['reward_mean'],
            'final_val_reward_mean': final_val_reward['reward_mean'],
        },
    )

    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('\nTraining complete.')
    print('Best val NLL: %.6f' % best_val_nll)
    print('Final train-seed reward: %.3f +- %.3f' % (final_train_reward['reward_mean'], final_train_reward['reward_std']))
    print('Final val-seed reward: %.3f +- %.3f' % (final_val_reward['reward_mean'], final_val_reward['reward_std']))
    print('Policy saved to: %s' % policy_path)
    print('Metrics saved to: %s' % metrics_path)
    print('Use for AIRL init: --init_policy_path %s' % policy_path)


if __name__ == '__main__':
    main()
