"""
Utility functions for manipulating Trainer.

(The primary reason these functions are here instead of in utils.py is to
prevent cyclic imports between imitation.trainer and imitation.util)
"""
from collections import defaultdict
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf

import imitation.discrim_net as discrim_net
from imitation.reward_net import BasicShapedRewardNet
from imitation.trainer import Trainer
import imitation.util as util


def init_trainer(env_id, rollouts_dir="data/rollouts", use_gail=False,
                 num_vec=8, discrim_scale=False,
                 n_rollout_dumps: int = 1,
                 discrim_kwargs={}, reward_kwargs={}, trainer_kwargs={},
                 make_blank_policy_kwargs={}):
  """Builds a Trainer, ready to be trained on a vectorized environment
  and expert demonstrations.

  Args:
    env_id (str): The string id of a gym environment.
    rollouts_dir (str): A directory containing .npz rollout files.
    use_gail (bool): If True, then train using GAIL. If False, then train
        using AIRL.
    policy_dir (str): The directory containing the pickled experts for
        generating rollouts.
    trainer_kwargs (dict): Aguments for the Trainer constructor.
    reward_kwargs (dict): Arguments for the `*RewardNet` constructor.
    discrim_kwargs (dict): Arguments for the `DiscrimNet*` constructor.
    n_rollout_dumps: The number of rollout .npz files to load as expert
        demonstrations.
    make_blank_policy_kwargs: Keyword arguments passed to `make_blank_policy`,
        used to initialize the trainer.
  """
  env = util.make_vec_env(env_id, num_vec)
  gen_policy = util.make_blank_policy(env, verbose=1,
                                      **make_blank_policy_kwargs)

  if use_gail:
    discrim = discrim_net.DiscrimNetGAIL(env.observation_space,
                                         env.action_space,
                                         scale=discrim_scale,
                                         **discrim_kwargs)
  else:
    rn = BasicShapedRewardNet(env.observation_space, env.action_space,
                              scale=discrim_scale, **reward_kwargs)
    discrim = discrim_net.DiscrimNetAIRL(rn, **discrim_kwargs)

  ro_paths = util.get_dump_paths(
    env,
    basedir=rollouts_dir,
    policy_class=gen_policy.__class__,
    n_dumps=n_rollout_dumps)
  expert_rollouts = _joined_expert_rollouts(ro_paths)
  trainer = Trainer(env, gen_policy, discrim, expert_rollouts,
                    **trainer_kwargs)
  return trainer


def _joined_expert_rollouts(npz_paths: Sequence[str],
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """From several .npz rollout dumps, generate rollout arrays for Trainer.

  Args:
      npz_paths: A listlike of paths to .npz files generated by the data_collect
          script.
  """
  shards = defaultdict(list)

  for path in npz_paths:
    with np.load(path) as npz:
      tf.logging.info(f"Loaded rollouts from '{path}'.")
      for k, v in npz.items():
        shards[k].append(v)

  shards = dict(shards)  # Convert to dict to error on missing keys.
  return tuple(np.concatenate(shards[key])
               for key in ["obs_old", "act", "obs_new"])
