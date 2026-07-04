#!/usr/bin/env python

import argparse
import csv
import os
import sys

import gym
import numpy as np
import safety_gym


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAFE_RL_ROOT = os.path.join(ROOT_DIR, '3rdparty', 'safety-starter-agents')
if SAFE_RL_ROOT not in sys.path:
    sys.path.append(SAFE_RL_ROOT)

from safe_rl.utils.load_utils import load_policy


DEFAULT_POLICY_DIR = os.path.join(
    ROOT_DIR,
    'airl',
    'data',
    '2026-03-16_airl_transfer_v4',
    '2026-03-16_18-10-04-airl_transfer_v4_s0',
    'resume_from_378_20260317_090451',
)


def safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return next_obs, reward, done, info
    return out


def evaluate(env, get_action, episodes, max_ep_len):
    returns, costs, lengths = [], [], []

    for _ in range(episodes):
        obs = safe_reset(env)
        ep_ret = 0.0
        ep_cost = 0.0
        ep_len = 0

        done = False
        while not done:
            action = get_action(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, done, info = safe_step(env, action)
            ep_ret += float(reward)
            ep_cost += float(info.get('cost', 0.0))
            ep_len += 1

            if max_ep_len > 0 and ep_len >= max_ep_len:
                break

        returns.append(ep_ret)
        costs.append(ep_cost)
        lengths.append(ep_len)

    return np.array(returns), np.array(costs), np.array(lengths)


def write_summary_csv(csv_path, env_id, policy_dir, itr, returns, costs, lengths):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['env_id', env_id])
        writer.writerow(['policy_dir', os.path.abspath(policy_dir)])
        writer.writerow(['itr', itr])
        writer.writerow(['episodes', int(len(returns))])
        writer.writerow(['return_mean', float(np.mean(returns))])
        writer.writerow(['return_std', float(np.std(returns))])
        writer.writerow(['return_min', float(np.min(returns))])
        writer.writerow(['return_max', float(np.max(returns))])
        writer.writerow(['cost_mean', float(np.mean(costs))])
        writer.writerow(['cost_std', float(np.std(costs))])
        writer.writerow(['cost_min', float(np.min(costs))])
        writer.writerow(['cost_max', float(np.max(costs))])
        writer.writerow(['ep_len_mean', float(np.mean(lengths))])
        writer.writerow(['ep_len_std', float(np.std(lengths))])
        writer.writerow(['ep_len_min', float(np.min(lengths))])
        writer.writerow(['ep_len_max', float(np.max(lengths))])


def main():
    parser = argparse.ArgumentParser(description='Evaluate exported policy and write summary CSV.')
    parser.add_argument('--policy_dir', type=str, default=DEFAULT_POLICY_DIR)
    parser.add_argument('--itr', type=int, default=399)
    parser.add_argument('--env_id', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument(
        '--out_csv',
        type=str,
        default=os.path.join(DEFAULT_POLICY_DIR, 'evaluation_summary.csv'),
    )
    args = parser.parse_args()

    itr_to_load = args.itr if args.itr >= 0 else 'last'
    env, get_action, sess = load_policy(args.policy_dir, itr=itr_to_load, deterministic=args.deterministic)

    if env is None:
        env = gym.make(args.env_id)

    returns, costs, lengths = evaluate(
        env=env,
        get_action=get_action,
        episodes=args.episodes,
        max_ep_len=args.max_ep_len,
    )

    write_summary_csv(
        csv_path=args.out_csv,
        env_id=args.env_id,
        policy_dir=args.policy_dir,
        itr=itr_to_load,
        returns=returns,
        costs=costs,
        lengths=lengths,
    )

    print('Evaluation complete')
    print('Policy dir: %s' % os.path.abspath(args.policy_dir))
    print('Loaded itr: %s' % str(itr_to_load))
    print('Episodes: %d' % args.episodes)
    print('Mean return: %.3f' % float(np.mean(returns)))
    print('Mean cost: %.3f' % float(np.mean(costs)))
    print('Summary CSV: %s' % os.path.abspath(args.out_csv))

    try:
        env.close()
    except Exception:
        pass
    try:
        sess.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
