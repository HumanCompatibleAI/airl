"""Utilities and definitions shared by reward-related code."""

import functools
import numpy as np
from typing import Callable, Optional
from stable_baselines.common import vec_env

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    np.ndarray]


def _reward_fn_normalize_inputs(obs: np.ndarray,
                                acts: np.ndarray,
                                next_obs: np.ndarray,
                                steps: Optional[np.ndarray] = None,
                                *,
                                reward_fn: RewardFn,
                                vec_normalize: vec_env.VecNormalize,
                                norm_reward: bool = True,
                                ) -> np.ndarray:
  """Combine with `functools.partial` to create an input-normalizing RewardFn.

  Args:
    reward_fn: The reward function that normalized inputs are evaluated on.
    vec_normalize: Instance of VecNormalize used to normalize inputs and
     rewards.
    norm_reward: If True, then also normalize reward before returning.
  Returns:
    The possibly normalized reward.
  """
  norm_obs = vec_normalize.norm_obs(obs)
  norm_next_obs = vec_normalize.norm_obs(next_obs)
  rew = reward_fn(norm_obs, acts, norm_next_obs, steps)
  if norm_reward:
    rew = vec_normalize.normalize_reward(rew)
  return rew


def build_norm_reward_fn(*, reward_fn, vec_normalize, **kwargs) -> RewardFn:
  """Reward function that automatically normalizes inputs.

  See _reward_fn_normalize_inputs for argument documentation.
  """
  return functools.partial(_reward_fn_normalize_inputs, reward_fn=reward_fn,
                           vec_normalize=vec_normalize, **kwargs)
