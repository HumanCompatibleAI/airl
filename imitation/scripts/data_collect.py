import tensorflow as tf

from imitation.scripts.config.data_collect import data_collect_ex
import imitation.util as util


def make_PPO2(env_name, num_vec):
  env = util.make_vec_env(env_name, num_vec)
  # TODO(adam): add support for wrapping env with VecNormalize
  # (This is non-trivial since we'd need to make sure it's also applied
  # when the policy is re-loaded to generate rollouts.)
  policy = util.make_blank_policy(env, verbose=1, init_tensorboard=True)
  return policy


@data_collect_ex.main
def main(env_name, total_timesteps, num_vec=8):
  tf.logging.set_verbosity(tf.logging.INFO)

  policy = make_PPO2(env_name, num_vec)

  callback = util.make_save_policy_callback("data/")
  policy.learn(total_timesteps, callback=callback)


if __name__ == "__main__":
    # TODO: Add observer
    data_collect_ex.run()
