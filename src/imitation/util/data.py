"""Types and helper methods for transitions and trajectories."""

import dataclasses
import os
import pickle
from typing import List, NamedTuple, Optional, Sequence

import numpy as np
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    infos: Optional[Sequence[dict]]
    """A sequence of info dicts, length trajectory_len."""

    def __len__(self):
        """Returns number of transitions, `trajectory_len` in attribute docstrings.

        This is equal to the number of actions. So this is zero when there is a
        single observation."""
        return len(self.acts)

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.obs) != len(self.acts) + 1:
            raise ValueError(
                "expected one more observation than actions: "
                f"{len(self.obs)} != {len(self.acts)} + 1"
            )
        if self.infos is not None and len(self.infos) != len(self.acts):
            raise ValueError(
                "infos when present must be present for each action:"
                f"{len(self.infos)} != {len(self.acts)}"
            )


def _rews_validation(rews: np.ndarray, acts: np.ndarray):
    if rews.shape != (len(acts),):
        raise ValueError(
            "rewards must be 1D array, one for each action:"
            f"{rews.shape} != ({len(acts)},)"
        )
    if rews.dtype not in [np.float32, np.float64, np.float128]:
        raise ValueError("rewards dtype {self.rews.dtype} not a float")


@dataclasses.dataclass(frozen=True)
class TrajectoryWithRew(Trajectory):
    rews: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation on shapes, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


@dataclasses.dataclass(frozen=True)
class Transitions:
    """A batch of obs-act-obs-done transitions.

    Often generated by combining and processing several TrajectoryNoRew objects via
    `flatten_trajectories()`.
    """

    obs: np.ndarray
    """
    Previous observations. Shape: (batch_size, ) + observation_shape.

    The i'th observation `obs[i]` in this array is the observation seen
    by the agent when choosing action `acts[i]`. obs.dtype == next_obs.dtype.
    """

    acts: np.ndarray
    """Actions. Shape: (batch_size,) + action_shape."""

    next_obs: np.ndarray
    """New observation. Shape: (batch_size, ) + observation_shape.

    The i'th observation `next_obs[i]` in this array is the observation
    after the agent has taken action `acts[i]`. next_obs.dtype == obs.dtype.
    """

    dones: np.ndarray
    """
    Boolean array indicating episode termination. Shape: (batch_size, ).

    `done[i]` is true iff `next_obs[i]` the last observation of an episode.
    """

    def __len__(self):
        """Returns number of transitions."""
        return len(self.obs)

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring."""
        if self.obs.shape != self.next_obs.shape:
            raise ValueError(
                "obs and next_obs must have same shape:"
                f"{self.obs.shape} != {self.next_obs.shape}"
            )
        if self.obs.dtype != self.next_obs.dtype:
            raise ValueError(
                "obs and next_obs must have the same dtype:"
                f"{self.obs.dtype} != {self.next_obs.dtype}"
            )
        if len(self.obs) != len(self.acts):
            raise ValueError(
                "obs and acts must have same number of timesteps:"
                f"{len(self.obs)} != {len(self.acts)}"
            )
        if self.dones.shape != (len(self.acts),):
            raise ValueError(
                "dones must be 1D array, one for each timestep:"
                f"{self.dones.shape} != ({len(self.acts)},)"
            )
        if self.dones.dtype != np.bool:
            raise ValueError(f"dones must be boolean, not {self.dones.dtype}")


@dataclasses.dataclass(frozen=True)
class TransitionsWithRew(Transitions):
    """A batch of obs-act-obs-rew-done transitions."""

    rews: np.ndarray
    """
    Reward. Shape: (batch_size, ). dtype float.

    The reward `rew[i]` at the i'th timestep is received after the
    agent has taken action `acts[i]`.
    """

    def __post_init__(self):
        """Performs input validation on shapes, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


class _TrajectoryBackwardCompatible(NamedTuple):
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    rews: np.ndarray
    """Reward, shape (trajectory_len, )."""

    infos: Optional[List[dict]]
    """A list of info dicts, length trajectory_len."""


def load(path: str) -> Sequence[TrajectoryWithRew]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # TODO(adam): remove backwards compatibility logic eventually (2021?)
    from unittest import mock

    with mock.patch(
        "imitation.util.rollout.Trajectory",
        new=_TrajectoryBackwardCompatible,
        create=True,
    ):
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
        if len(trajectories) > 0:
            if isinstance(trajectories[0], _TrajectoryBackwardCompatible):
                trajectories = [
                    TrajectoryWithRew(**traj._asdict()) for traj in trajectories
                ]
    return trajectories


def save(path: str, trajectories: Sequence[TrajectoryWithRew]) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(trajectories, f)
    # Ensure atomic write
    os.replace(path + ".tmp", path)
    tf.logging.info("Dumped demonstrations to {}.".format(path))
