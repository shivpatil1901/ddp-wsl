import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import gym
import safety_gym
import safe_rl
import os
import shutil
import wandb
from safe_rl.utils.run_utils import setup_logger_kwargs

# --- 1. THE WRAPPER (Capturing True Rewards) ---
class LearnedRewardWrapper(gym.Wrapper):
    def __init__(self, env, discriminator_tensors, sess, obs_dim=60, act_dim=2):
        super().__init__(env)
        self.sess = sess
        self.input_ph, self.g_output = discriminator_tensors

    def step(self, action):
        # 1. Get ground truth reward
        obs, true_reward, done, info = self.env.step(action)
        
        # 2. Store true_reward in info so the logger can find it
        info['true_reward'] = true_reward
        
        # 3. Calculate Learned Reward
        sa_input = np.concatenate([obs.reshape(1, -1), action.reshape(1, -1)], axis=1)
        learned_reward = self.sess.run(self.g_output, feed_dict={self.input_ph: sa_input})[0][0]
        
        return obs, learned_reward, done, info


def _extract_itr_from_simple_save(dirname):
    suffix = dirname[len('simple_save'):]
    return int(suffix) if suffix.isdigit() else -1


def _export_latest_policy_artifacts(output_dir):
    """Create a stable, reloadable export from the latest simple_save* checkpoint."""
    simple_save_dirs = [
        d for d in os.listdir(output_dir)
        if d.startswith('simple_save') and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not simple_save_dirs:
        raise RuntimeError(f"No simple_save* directories found in {output_dir}")

    latest_name = max(simple_save_dirs, key=_extract_itr_from_simple_save)
    latest_itr = _extract_itr_from_simple_save(latest_name)
    latest_dir = os.path.join(output_dir, latest_name)

    export_dir = os.path.join(output_dir, 'policy_final')
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    shutil.copytree(latest_dir, export_dir)

    vars_src = os.path.join(output_dir, f'vars{latest_itr}.pkl')
    vars_dst = os.path.join(output_dir, 'policy_final_vars.pkl')
    if os.path.exists(vars_src):
        shutil.copy2(vars_src, vars_dst)

    return latest_itr, export_dir

# --- 2. MAIN TRAINING FUNCTION ---
def main(checkpoint_path, seed=0):
    robot, task = 'Point', 'Goal1'
    obs_dim, act_dim = 60, 2
    cost_lim = 25.0
    steps_per_epoch = 30000
    epochs = 330

    wandb.init(project='safe_airl_final', name=f'run_s{seed}')

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    
    # --- ROBUST LOADING BLOCK ---
    # We define the variables exactly as the error message showed they expect
    input_ph = tf.placeholder(tf.float32, shape=(None, obs_dim + act_dim), name='sa_input')
    
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # Match the architecture exactly
        x = tf.layers.dense(input_ph, 256, activation=tf.nn.relu, name='dense')
        x = tf.layers.dense(x, 256, activation=tf.nn.relu, name='dense_1')
        g_output = tf.layers.dense(x, 1, name='dense_2')

    # Initialize all variables first
    sess.run(tf.global_variables_initializer())

    # Map variables from the checkpoint to our graph
    # Your error shows variables are named 'dense/kernel', 'dense/bias', etc.
    var_map = {
        'dense/kernel': tf.get_default_graph().get_tensor_by_name('dense/kernel:0'),
        'dense/bias': tf.get_default_graph().get_tensor_by_name('dense/bias:0'),
        'dense_1/kernel': tf.get_default_graph().get_tensor_by_name('dense_1/kernel:0'),
        'dense_1/bias': tf.get_default_graph().get_tensor_by_name('dense_1/bias:0'),
        'dense_2/kernel': tf.get_default_graph().get_tensor_by_name('dense_2/kernel:0'),
        'dense_2/bias': tf.get_default_graph().get_tensor_by_name('dense_2/bias:0'),
    }

    try:
        saver_dl = tf.train.Saver(var_list=var_map)
        saver_dl.restore(sess, checkpoint_path)
        print("\n[SUCCESS] Discriminator weights restored via TF Saver.\n")
    except Exception as e:
        print(f"\n[WARNING] Standard Restore failed: {e}. Attempting Keras-style fallback...")
        # If the above names aren't in the CKPT, this allows a flexible load
        loader = tf.train.Saver()
        loader.restore(sess, checkpoint_path)

    # --- LOGGER SETUP ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    logger_kwargs = setup_logger_kwargs('airl_transfer_v4', seed, data_dir=data_dir)

    def env_fn():
        base_env = gym.make(f'Safexp-{robot}{task}-v0')
        return LearnedRewardWrapper(base_env, (input_ph, g_output), sess)

    # 3. Launch Training
    safe_rl.ppo_lagrangian(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(256, 256)),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        cost_lim=cost_lim,
        seed=seed,
        logger_kwargs=logger_kwargs
    )

    # --- 4. EXPORT RELOADABLE POLICY ---
    latest_itr, export_dir = _export_latest_policy_artifacts(logger_kwargs['output_dir'])
    print(f"\n[SUCCESS] Learned policy exported to: {export_dir}")
    print(f"[INFO] Reload with: python test_policy.py {logger_kwargs['output_dir']} --itr={latest_itr}")

    wandb.finish()

if __name__ == '__main__':
    # Full path to your checkpoint
    CHECKPOINT = "./airl_results/reward_model_final.ckpt"
    main(CHECKPOINT)