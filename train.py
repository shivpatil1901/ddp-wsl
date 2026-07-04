#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl.utils.mpi_tools import proc_id
import time
import os.path as osp
import wandb


def main(robot, task, algo, seed, cost_lim, num_cpus):
    # Verify experiment
    robot_list = ['point', 'car']  # ['doggo']
    task_list = ['goal1', 'button1', 'push1',]  # ['goal2', 'button2', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian']  # ['trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 1  # Save at every 10 epochs
    target_kl = 0.01

    # Fork for parallelizing
    mpi_fork(num_cpus)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir='./data', datestamp=False)

    if proc_id() == 0:
        wandb.init(
            project='saferl',
            name=f'{exp_name}_s{seed}',
            config={
                'robot': robot,
                'task': task,
                'algo': algo,
                'seed': seed,
                'epochs': epochs,
                'save_freq': save_freq,
                'target_kl': target_kl,
                'cost_lim': cost_lim,
            }
        )

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    algo(
        env_fn=lambda: gym.make(env_name),
        ac_kwargs=dict(hidden_sizes=(256, 256),),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        save_freq=save_freq,
        target_kl=target_kl,
        cost_lim=cost_lim,
        seed=seed,
        logger_kwargs=logger_kwargs
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')  # ppo, ppo_lagrangian
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=10)  # The number of processes
    parser.add_argument('--cost_lim', type=float, default=25)  # The number of processes
    args = parser.parse_args()
    main(args.robot, args.task, args.algo, args.seed, args.cost_lim, args.cpu)
