import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import gym
import safety_gym
import safe_rl
import os
import time
import shutil
import wandb

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


def _load_discriminator(sess, input_ph, checkpoint_path):
    """Load discriminator weights into the reward network graph."""
    var_map = {
        'dense/kernel': tf.get_default_graph().get_tensor_by_name('dense/kernel:0'),
        'dense/bias': tf.get_default_graph().get_tensor_by_name('dense/bias:0'),
        'dense_1/kernel': tf.get_default_graph().get_tensor_by_name('dense_1/kernel:0'),
        'dense_1/bias': tf.get_default_graph().get_tensor_by_name('dense_1/bias:0'),
        'dense_2/kernel': tf.get_default_graph().get_tensor_by_name('dense_2/kernel:0'),
        'dense_2/bias': tf.get_default_graph().get_tensor_by_name('dense_2/bias:0'),
    }
    saver_dl = tf.train.Saver(var_list=var_map)
    saver_dl.restore(sess, checkpoint_path)


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


def _copy_discriminator_checkpoint(src_ckpt_prefix, dst_dir):
    """Copy discriminator checkpoint files (*.index/*.data) into output dir."""
    copied = []
    for suffix in ['.index', '.data-00000-of-00001']:
        src = src_ckpt_prefix + suffix
        if os.path.exists(src):
            dst = os.path.join(dst_dir, 'reward_model_final.ckpt' + suffix)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def resume_training_simplified(resume_dir, resume_epoch, seed=0, total_epochs=400, discriminator_checkpoint=None):
    """
    Resume training by continuing from the last saved epoch.
    
    This approach:
    1. Loads the learned reward model from the discriminator checkpoint
    2. Lets safe_rl create a fresh training graph
    3. The logger will automatically append to progress.txt
    
    Args:
        resume_dir: Directory containing the training checkpoint 
        resume_epoch: Which epoch to resume from (e.g., 378)
        seed: Seed value
    """
    
    robot, task = 'Point', 'Goal1'
    obs_dim, act_dim = 60, 2
    cost_lim = 25.0
    steps_per_epoch = 30000
    next_epoch = resume_epoch + 1
    remaining_epochs = total_epochs - next_epoch

    if remaining_epochs <= 0:
        raise ValueError("No epochs remaining. Check total_epochs and resume_epoch values.")

    print(f"\n{'='*80}")
    print(f"RESUMING TRAINING FROM CHECKPOINT EPOCH {resume_epoch}")
    print(f"Total epochs to train: {total_epochs}")
    print(f"Remaining epochs: {remaining_epochs} (epochs {next_epoch} to {total_epochs - 1})")
    print(f"Source run directory: {resume_dir}")
    print(f"{'='*80}\n")

    # Verify that checkpoint exists
    checkpoint_dir = os.path.join(resume_dir, f'simple_save{resume_epoch}')
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if discriminator_checkpoint is None:
        discriminator_checkpoint = os.path.join(script_dir, 'airl_results', 'reward_model_final.ckpt')
    elif not os.path.isabs(discriminator_checkpoint):
        discriminator_checkpoint = os.path.abspath(discriminator_checkpoint)

    if not os.path.exists(discriminator_checkpoint + '.index'):
        raise ValueError(f"Discriminator checkpoint not found: {discriminator_checkpoint}")

    resume_stamp = time.strftime('%Y%m%d_%H%M%S')
    resume_output_dir = os.path.join(resume_dir, f'resume_from_{resume_epoch}_{resume_stamp}')
    os.makedirs(resume_output_dir, exist_ok=False)
    
    print(f"[OK] Found checkpoint: {checkpoint_dir}\n")
    print(f"[OK] Found discriminator checkpoint: {discriminator_checkpoint}\n")
    print(f"[OK] Resume output directory: {resume_output_dir}\n")

    wandb.init(
        project='safe_airl_final', 
        name=f'resume_s{seed}_from_epoch{resume_epoch}',
        resume='allow'
    )

    # Setup TensorFlow session for discriminator
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    
    # --- Setup discriminator reward model ---
    input_ph = tf.placeholder(tf.float32, shape=(None, obs_dim + act_dim), name='sa_input')
    
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(input_ph, 256, activation=tf.nn.relu, name='dense')
        x = tf.layers.dense(x, 256, activation=tf.nn.relu, name='dense_1')
        g_output = tf.layers.dense(x, 1, name='dense_2')

    sess.run(tf.global_variables_initializer())
    _load_discriminator(sess, input_ph, discriminator_checkpoint)
    print("[OK] Discriminator restored\n")

    def env_fn():
        base_env = gym.make(f'Safexp-{robot}{task}-v0')
        return LearnedRewardWrapper(base_env, (input_ph, g_output), sess)

    # --- Train the remaining epochs, restoring policy weights from last checkpoint ---
    policy_restore_path = os.path.join(resume_dir, f'simple_save{resume_epoch}')
    print(f"[INFO] Restoring policy weights from: {policy_restore_path}")
    print(f"[INFO] Starting training for {remaining_epochs} epochs with epoch offset {next_epoch}...\n")
    
    safe_rl.ppo_lagrangian(
        env_fn=env_fn,
        ac_kwargs=dict(hidden_sizes=(256, 256)),
        epochs=remaining_epochs,  # Only remaining epochs!
        steps_per_epoch=steps_per_epoch,
        cost_lim=cost_lim,
        seed=seed,
        logger_kwargs=dict(output_dir=resume_output_dir, exp_name='airl_transfer_v4_resume'),
        save_freq=1,
        restore_path=policy_restore_path,
        start_epoch=next_epoch
    )

    latest_itr, export_dir = _export_latest_policy_artifacts(resume_output_dir)
    copied_ckpts = _copy_discriminator_checkpoint(discriminator_checkpoint, resume_output_dir)

    print(f"\n{'='*80}")
    print(f"✓ TRAINING COMPLETE!")
    print(f"Final checkpoint: {resume_output_dir}/simple_save{total_epochs-1}")
    print(f"Exported final policy: {export_dir}")
    print(f"Exported policy iteration: {latest_itr}")
    if copied_ckpts:
        print(f"Copied discriminator checkpoint to: {resume_output_dir}/reward_model_final.ckpt*")
    print(f"Progress logged to: {resume_output_dir}/progress.txt")
    print(f"{'='*80}\n")

    wandb.finish()
    sess.close()


if __name__ == '__main__':
    # Configure these paths
    RESUME_DIRECTORY = "./airl/data/2026-03-16_airl_transfer_v4/2026-03-16_18-10-04-airl_transfer_v4_s0"
    RESUME_FROM_EPOCH = 378
    SEED = 0
    TOTAL_EPOCHS = 400
    DISCRIMINATOR_CHECKPOINT = "./airl/airl_results/reward_model_final.ckpt"

    resume_training_simplified(
        resume_dir=RESUME_DIRECTORY,
        resume_epoch=RESUME_FROM_EPOCH,
        seed=SEED,
        total_epochs=TOTAL_EPOCHS,
        discriminator_checkpoint=DISCRIMINATOR_CHECKPOINT,
    )
