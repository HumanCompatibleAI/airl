import collections
import functools
import glob
import os
import pickle
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Union

import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import VecEnv
import tensorflow as tf

from imitation.policies.base import get_action_policy


class Trajectory(NamedTuple):
  """A trajectory, e.g. a one episode rollout from an expert policy.

   Attributes:
    obs: Observations, shape (trajectory_len+1, ) + observation_shape.
    act: Actions, shape (trajectory_len, ) + action_shape.
    rew: Reward, shape (trajectory_len, ).
    infos: A list of info dicts, length (trajectory_len, ).
  """

  # TODO(shwang): IN this and Transitions, act=>acts, rew=>rews (matches
  # Baselines). done=>dones
  act: np.ndarray
  obs: np.ndarray
  rew: np.ndarray
  infos: List[dict]


class Transitions(NamedTuple):
  """A batch of obs-act-obs-rew-done transitions.

  Usually generated by combining and processing several Trajectories via
  `flatten_trajectory()`.

  Attributes:
    obs: Previous observations. Shape: (batch_size, ) + observation_shape.
        The i'th observation `obs[i]` in this array is the observation seen
        by the agent when choosing action `act[i]`.
    act: Actions. Shape: (batch_size, ) + action_shape.
    next_obs: New observation. Shape: (batch_size, ) + observation_shape.
        The i'th observation `next_obs[i]` in this array is the observation
        after the agent has taken action `act[i]`.
    rew: Reward. Shape: (batch_size, ).
        The reward `rew[i]` at the i'th timestep is received after the agent has
        taken action `act[i]`.
    done: Boolean array indicating episode termination. Shape: (batch_size, ).
        `done[i]` is true iff `next_obs[i]` the last observation of an episode.
  """

  obs: np.ndarray
  act: np.ndarray
  next_obs: np.ndarray
  rew: np.ndarray
  done: np.ndarray


class _TrajectoryAccumulator:
  """Accumulates trajectories step-by-step.

  Used in `generate_trajectories()` only, for collecting completed trajectories
  while ignoring partially-completed trajectories.
  """

  def __init__(self):
    self.partial_trajectories = collections.defaultdict(list)

  def finish_trajectory(self, idx) -> Trajectory:
    """Complete the trajectory labelled with `idx`.

    Return list of completed trajectories popped from
    `self.partial_trajectories`.
    """
    part_dicts = self.partial_trajectories[idx]
    del self.partial_trajectories[idx]
    out_dict_unstacked = collections.defaultdict(list)
    for part_dict in part_dicts:
      for key, array in part_dict.items():
        out_dict_unstacked[key].append(array)
    out_dict_stacked = {
        key: np.stack(arr_list, axis=0)
        for key, arr_list in out_dict_unstacked.items()
    }
    return Trajectory(**out_dict_stacked)

  def add_step(self, idx, step_dict: Dict[str, np.ndarray]):
    """Add a single step to the partial trajectory identified by `idx`.

    This could correspond to, e.g., one environment managed by a VecEnv.
    """
    self.partial_trajectories[idx].append(step_dict)


GenTrajTerminationFn = Callable[[Sequence[Trajectory]], bool]


def min_episodes(n: int) -> GenTrajTerminationFn:
  """Terminate after collecting n episodes of data.

  Argument:
    n: Minimum number of episodes of data to collect.
        May overshoot if two episodes complete simultaneously (unlikely).

  Returns:
    A function implementing this termination condition.
  """
  def f(trajectories: Sequence[Trajectory]):
    return len(trajectories) >= n
  return f


def min_timesteps(n: int) -> GenTrajTerminationFn:
  """Terminate at the first episode after collecting n timesteps of data.

  Arguments:
    n: Minimum number of timesteps of data to collect.
        May overshoot to nearest episode boundary.

  Returns:
    A function implementing this termination condition.
  """
  def f(trajectories: Sequence[Trajectory]):
    timesteps = sum(len(t.obs) - 1 for t in trajectories)
    return timesteps >= n
  return f


def make_sample_until(n_timesteps: Optional[int],
                      n_episodes: Optional[int],
                      ) -> GenTrajTerminationFn:
  """Returns a termination condition sampling until n_timesteps or n_episodes.

  Arguments:
    n_timesteps: Minimum number of timesteps to sample.
    n_episodes: Number of episodes to sample.

  Returns:
    A termination condition.

  Raises:
    ValueError if both or neither of n_timesteps and n_episodes are set,
    or if either are non-positive.
  """
  if n_timesteps is not None and n_episodes is not None:
    raise ValueError("n_timesteps and n_episodes were both set")
  elif n_timesteps is not None:
    assert n_timesteps > 0
    return min_timesteps(n_timesteps)
  elif n_episodes is not None:
    assert n_episodes > 0
    return min_episodes(n_episodes)
  else:
    raise ValueError("Set at least one of n_timesteps and n_episodes")


def generate_trajectories(policy,
                          venv: VecEnv,
                          sample_until: GenTrajTerminationFn,
                          *,
                          deterministic_policy: bool = False,
                          ) -> Sequence[Trajectory]:
  """Generate trajectory dictionaries from a policy and an environment.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    venv: The vectorized environments to interact with.
    sample_until: A function determining the termination condition.
        It takes a sequence of trajectories, and returns a bool.
        Most users will want to use one of `min_episodes` or `min_timesteps`.
    deterministic_policy: If True, asks policy to deterministically return
        action. Note the trajectories might still be non-deterministic if the
        environment has non-determinism!

  Returns:
    Sequence of `Trajectory` named tuples.
  """
  if isinstance(policy, BaseRLModel):
    get_action = policy.predict
    policy.set_env(venv)
  else:
    get_action = functools.partial(get_action_policy, policy)

  # Collect rollout tuples.
  trajectories = []
  # accumulator for incomplete trajectories
  trajectories_accum = _TrajectoryAccumulator()
  obs_batch = venv.reset()
  for env_idx, obs in enumerate(obs_batch):
    # Seed with first obs only. Inside loop, we'll only add second obs from
    # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
    # get all observations, but they're not duplicated into "next obs" and
    # "previous obs" (this matters for, e.g., Atari, where observations are
    # really big).
    trajectories_accum.add_step(env_idx, dict(obs=obs))
  while not sample_until(trajectories):
    obs_old_batch = obs_batch
    act_batch, _ = get_action(obs_old_batch, deterministic=deterministic_policy)
    obs_batch, rew_batch, done_batch, info_batch = venv.step(act_batch)

    # Don't save tuples if there is a done. The next_obs for any environment
    # is incorrect for any timestep where there is an episode end, so we fix it
    # with returned state info.
    zip_iter = enumerate(
        zip(obs_old_batch, act_batch, obs_batch, rew_batch, done_batch,
            info_batch))
    for env_idx, (obs_old, act, obs, rew, done, info) in zip_iter:
      real_obs = obs
      if done:
        # actual obs is inaccurate, so we use the one inserted into step info
        # by stable baselines wrapper
        real_obs = info['terminal_observation']
      trajectories_accum.add_step(
          env_idx,
          dict(
              act=act,
              rew=rew,
              # this is not the obs corresponding to `act`, but rather the obs
              # *after* `act` (see above)
              obs=real_obs,
              infos=info))
      if done:
        # finish env_idx-th trajectory
        new_traj = trajectories_accum.finish_trajectory(env_idx)
        trajectories.append(new_traj)
        trajectories_accum.add_step(env_idx, dict(obs=obs))
        continue

  # Note that we just drop partial trajectories. This is not ideal for some
  # algos; e.g. BC can probably benefit from partial trajectories, too.

  # Sanity checks.
  for trajectory in trajectories:
    n_steps = len(trajectory.act)
    # extra 1 for the end
    exp_obs = (n_steps + 1, ) + venv.observation_space.shape
    real_obs = trajectory.obs.shape
    assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
    exp_act = (n_steps, ) + venv.action_space.shape
    real_act = trajectory.act.shape
    assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
    exp_rew = (n_steps,)
    real_rew = trajectory.rew.shape
    assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

  return trajectories


def rollout_stats(policy, venv: VecEnv, sample_until: GenTrajTerminationFn,
                  **kwargs):
  """Rolls out trajectories under the policy and returns various statistics.

  Args:
      policy (stable_baselines.BasePolicy): A stable_baselines Model,
          trained on the gym environment.
      venv: The vectorized environment to interact with.
      n_timesteps (int): The number of rewards to collect.
      n_episodes (int): The minimum number of episodes to finish before we stop
          collecting rewards. Rewards from parallel episodes that are underway
          when the final episode is finished are also included in the return.
      **kwargs: passed through to `generate_trajectories`.

  Returns:
      Dictionary containing `n_traj` collected (int), along with episode return
      statistics (keys: `{monitor_,}return_{min,mean,std,max}`, float values)
      and trajectory length statistics (keys: `len_{min,mean,std,max}`, float
      values).

      `return_*` values are calculated from environment rewards.
      `monitor_return_*` values are calculated using the `infos['epinfo']['r']`
      rewards generated by a Monitor wrapper (this can be useful for bypassing
      wrappers that modify the reward).
  """
  trajectories = generate_trajectories(policy, venv, sample_until, **kwargs)
  assert len(trajectories) > 0
  out_stats = {"n_traj": len(trajectories)}
  traj_descriptors = {
    "return": np.asarray([sum(t.rew) for t in trajectories]),
    "len": np.asarray([len(t.rew) for t in trajectories]),
  }
  if "episode" in trajectories[0].infos[-1]:
    monitor_ep_returns = [t.infos[-1]["episode"]["r"] for t in trajectories]
    traj_descriptors["monitor_return"] = np.asarray(monitor_ep_returns)

  stat_names = ["min", "mean", "std", "max"]
  for desc_name, desc_vals in traj_descriptors.items():
    for stat_name in stat_names:
      stat_value = getattr(np, stat_name)(desc_vals)
      out_stats[f"{desc_name}_{stat_name}"] = stat_value
  return out_stats


def mean_return(*args, **kwargs) -> float:
  """Find the mean return of a policy.

  Shortcut to call `rollout_stats` and fetch only the value for
  `return_mean`; see documentation for `rollout_stats`.
  """
  return rollout_stats(*args, **kwargs)["return_mean"]


def flatten_trajectories(trajectories: Sequence[Trajectory]) -> Transitions:
  """Flatten a series of trajectory dictionaries into arrays.

  Returns observations, actions, next observations, rewards.

  Args:
      trajectories: list of trajectories.

  Returns:
    The trajectories flattened into a single batch of Transitions.
  """
  keys = ["obs", "next_obs", "act", "rew", "done"]
  parts = {key: [] for key in keys}
  for traj in trajectories:
    parts["act"].append(traj.act)
    parts["rew"].append(traj.rew)
    obs = traj.obs
    parts["obs"].append(obs[:-1])
    parts["next_obs"].append(obs[1:])
    done = np.zeros_like(traj.rew, dtype=np.bool)
    done[-1] = True
    parts["done"].append(done)
  cat_parts = {
    key: np.concatenate(part_list, axis=0)
    for key, part_list in parts.items()
  }
  lengths = set(map(len, cat_parts.values()))
  assert len(lengths) == 1, f"expected one length, got {lengths}"
  return Transitions(**cat_parts)


def generate_transitions(policy,
                         venv,
                         n_timesteps: int,
                         *,
                         truncate: bool = True,
                         **kwargs) -> Transitions:
  """Generate obs-action-next_obs-reward tuples.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    venv: The vectorized environments to interact with.
    n_timesteps: The minimum number of timesteps to sample.
    truncate: If True, then drop any additional samples to ensure that exactly
        `n_timesteps` samples are returned.
    **kwargs: Passed-through to generate_trajectories.

  Returns:
    A batch of Transitions. The length of the constituent arrays is guaranteed
    to be at least `n_timesteps` (if specified), but may be greater unless
    `truncate` is provided as we collect data until the end of each episode.
  """
  traj = generate_trajectories(policy, venv,
                               sample_until=min_timesteps(n_timesteps),
                               **kwargs)
  transitions = flatten_trajectories(traj)
  if truncate and n_timesteps is not None:
    transitions = Transitions(*(arr[:n_timesteps] for arr in transitions))
  return transitions


def save(path: str,
         policy: Union[BaseRLModel, BasePolicy],
         venv: VecEnv,
         sample_until: GenTrajTerminationFn,
         **kwargs,
         ) -> None:
    """Generate policy rollouts and save them to a pickled Sequence[Trajectory].

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      deterministic_policy: Argument from `generate_trajectories`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trajs = generate_trajectories(policy, venv, sample_until, **kwargs)
    with open(path, "wb") as f:
      pickle.dump(trajs, f)
    tf.logging.info("Dumped demonstrations to {}.".format(path))


def load_trajectories(rollout_glob: str,
                      max_n_files: Optional[int] = None,
                      ) -> Sequence[Trajectory]:
  """Load trajectories from rollout pickles.

  Args:
      rollout_glob: Glob path to rollout pickles.
      max_n_files: If provided, then only load the most recent `max_n_files`
          files, as sorted by modification times.

  Returns:
      A list of trajectory dictionaries.

  Raises:
      ValueError: No files match the glob.
  """
  ro_paths = glob.glob(rollout_glob)
  if len(ro_paths) == 0:
    raise ValueError(f"No files match glob '{rollout_glob}'")
  if max_n_files is not None:
    ro_paths.sort(key=os.path.getmtime)
    ro_paths = ro_paths[-max_n_files:]

  traj_joined = []  # type: List[Trajectory]
  for path in ro_paths:
    with open(path, "rb") as f:
      traj = pickle.load(f)  # type: Sequence[Trajectory]
      tf.logging.info(f"Loaded rollouts from '{path}'.")
      traj_joined.extend(traj)

  return traj_joined
