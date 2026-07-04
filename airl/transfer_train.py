import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym 
import safety_gym
import safe_rl
import logging
import os
import wandb # Added back for proper logging
from tqdm import tqdm
from safe_rl.utils.run_utils import setup_logger_kwargs

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("transfer_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AIRLDiscriminator(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_size=256, gamma=0.99):
        super(AIRLDiscriminator, self).__init__()
        self.gamma = gamma
        self.g_network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(state_dim + action_dim,)),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.h_network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(state_dim,)),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
    def call(self, states, actions, next_states):
        sa = tf.concat([states, actions], axis=-1)
        g_sa = self.g_network(sa)
        h_s = self.h_network(states)
        h_s_next = self.h_network(next_states)
        return g_sa + self.gamma * h_s_next - h_s

class LearnedRewardWrapper(gym.Wrapper):
    def __init__(self, env, discriminator, sess, obs_dim=60, act_dim=2):
        super().__init__(env)
        self.discriminator = discriminator
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_ph = tf.placeholder(tf.float32, shape=(None, obs_dim + act_dim))
        self.g_output = self.discriminator.g_network(self.input_ph)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Dimension fix: Reshape to ensure both are (1, N)
        obs_reshaped = obs.reshape(1, -1)
        act_reshaped = action.reshape(1, -1)
        sa_input = np.concatenate([obs_reshaped, act_reshaped], axis=1)
        
        # Use TF 1.13.1 Session to get the reward
        learned_reward = self.sess.run(self.g_output, 
                                     feed_dict={self.input_ph: sa_input})[0][0]
        
        return obs, learned_reward, done, info

def main(checkpoint_path, seed=0):
    robot, task = 'Point', 'Goal1'
    obs_dim, act_dim = 60, 2
    cost_lim = 25.0
    num_steps = 1e7
    steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)

    # Initialize WandB session (Fixes the preinit error)
    wandb.init(
        project='airl_reward_transfer',
        name=f'transfer_{robot}_{task}_s{seed}',
        config={
            "algo": "ppo_lagrangian",
            "cost_lim": cost_lim,
            "epochs": epochs,
            "seed": seed
        }
    )

    # Initialize TF 1.13.1 Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    discriminator = AIRLDiscriminator(obs_dim, act_dim)
    sess.run(tf.global_variables_initializer())
    discriminator.load_weights(checkpoint_path)
    logger.info("Weights loaded. Training starting...")

    exp_name = f'airl_transfer_{robot.lower()}{task.lower()}'
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir='./data')

    def env_fn():
        base_env = gym.make(f'Safexp-{robot}{task}-v0')
        return LearnedRewardWrapper(base_env, discriminator, sess, obs_dim, act_dim)

    # Launch PPO-Lagrangian
    safe_rl.ppo_lagrangian(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(256, 256)),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        cost_lim=cost_lim,
        seed=seed, 
        logger_kwargs=logger_kwargs
    )
    
    wandb.finish()

if __name__ == '__main__':
    CHECKPOINT = "./airl_results/reward_model_final.ckpt"
    main(CHECKPOINT)