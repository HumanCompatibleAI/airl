"""Types and helper methods for transitions and trajectories."""

import dataclasses
import logging
import os
import pathlib
import pickle
from typing import Dict, Mapping, Optional, Sequence, TypeVar, Union, overload

import numpy as np
import torch as th
from torch.utils import data as th_data

from imitation.data import old_types

T = TypeVar("T")

AnyPath = Union[str, bytes, os.PathLike]


def dataclass_quick_asdict(dataclass_instance) -> dict:
    """Extract dataclass to items using `dataclasses.fields` + dict comprehension.

    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.
    """
    obj = dataclass_instance
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """A trajectory, e.g. a one episode rollout from an expert policy."""

    obs: np.ndarray
    """Observations, shape (trajectory_len + 1, ) + observation_shape."""

    acts: np.ndarray
    """Actions, shape (trajectory_len, ) + action_shape."""

    infos: Optional[np.ndarray]
    """An array of info dicts, length trajectory_len."""

    def __len__(self):
        """Returns number of transitions, `trajectory_len` in attribute docstrings.

        This is equal to the number of actions, and is always positive.
        """
        return len(self.acts)

    def __post_init__(self):
        """Performs input validation: check shapes are as specified in docstring."""
        if len(self.obs) != len(self.acts) + 1:
            raise ValueError(
                "expected one more observations than actions: "
                f"{len(self.obs)} != {len(self.acts)} + 1"
            )
        if self.infos is not None and len(self.infos) != len(self.acts):
            raise ValueError(
                "infos when present must be present for each action: "
                f"{len(self.infos)} != {len(self.acts)}"
            )
        if len(self.acts) == 0:
            raise ValueError("Degenerate trajectory: must have at least one action.")


def _rews_validation(rews: np.ndarray, acts: np.ndarray):
    if rews.shape != (len(acts),):
        raise ValueError(
            "rewards must be 1D array, one entry for each action: "
            f"{rews.shape} != ({len(acts)},)"
        )
    if not np.issubdtype(rews.dtype, np.floating):
        raise ValueError(f"rewards dtype {rews.dtype} not a float")


@dataclasses.dataclass(frozen=True)
class TrajectoryWithRew(Trajectory):
    rews: np.ndarray
    """Reward, shape (trajectory_len, ). dtype float."""

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


def transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, Union[np.ndarray, th.Tensor]]:
    """Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.

    Use this as the `collate_fn` argument to `DataLoader` if using an instance of
    `TransitionsMinimal` as the `dataset` argument.

    Handles all collation except "infos" collation using Torch's default collate_fn.
    "infos" needs special handling because we shouldn't recursively collate every
    the info dict into a single dict, but instead join all the info dicts into a list of
    dicts.
    """
    batch_no_infos = [
        {k: v for k, v in sample.items() if k != "infos"} for sample in batch
    ]

    result = th_data.dataloader.default_collate(batch_no_infos)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample in batch]
    return result


@dataclasses.dataclass(frozen=True)
class TransitionsMinimal(th_data.Dataset):
    """A Torch-compatible `Dataset` of obs-act transitions.

    This class and its subclasses are usually instantiated via
    `imitation.data.rollout.flatten_trajectories`.

    Indexing an instance `trans` of TransitionsMinimal with an integer `i`
    returns the `i`th `Dict[str, np.ndarray]` sample, whose keys are the field
    names of each dataclass field and whose values are the ith elements of each field
    value.

    Slicing returns a possibly empty instance of `TransitionsMinimal` where each
    field has been sliced.
    """

    obs: np.ndarray
    """
    Previous observations. Shape: (batch_size, ) + observation_shape.

    The i'th observation `obs[i]` in this array is the observation seen
    by the agent when choosing action `acts[i]`. `obs[i]` is not required to
    be from the timestep preceding `obs[i+1]`.
    """

    acts: np.ndarray
    """Actions. Shape: (batch_size,) + action_shape."""

    infos: np.ndarray
    """Array of info dicts. Shape: (batch_size,)."""

    def __len__(self):
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring.

        Also make array values read-only.
        """
        for val in vars(self).values():
            if isinstance(val, np.ndarray):
                val.setflags(write=False)

        if len(self.obs) != len(self.acts):
            raise ValueError(
                "obs and acts must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.acts)}"
            )

        if self.infos is not None and len(self.infos) != len(self.obs):
            raise ValueError(
                "obs and infos must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.infos)}"
            )

    @overload
    def __getitem__(self: T, key: slice) -> T:
        pass  # pragma: no cover

    @overload
    def __getitem__(self, key: int) -> Dict[str, np.ndarray]:
        pass  # pragma: no cover

    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics."""
        d = dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items()}

        if isinstance(key, slice):
            # Return type is the same as this dataclass. Replace field value with
            # slices.
            return dataclasses.replace(self, **d_item)
        else:
            assert isinstance(key, int)
            # Return type is a dictionary. Array values have no batch dimension.
            #
            # Dictionary of np.ndarray values is a convenient
            # torch.util.data.Dataset return type, as a torch.util.data.DataLoader
            # taking in this `Dataset` as its first argument knows how to
            # automatically concatenate several dictionaries together to make
            # a single dictionary batch with `torch.Tensor` values.
            return d_item


@dataclasses.dataclass(frozen=True)
class Transitions(TransitionsMinimal):
    """A batch of obs-act-obs-done transitions."""

    next_obs: np.ndarray
    """New observation. Shape: (batch_size, ) + observation_shape.

    The i'th observation `next_obs[i]` in this array is the observation
    after the agent has taken action `acts[i]`.

    Invariants:
        * `next_obs.dtype == obs.dtype`
        * `len(next_obs) == len(obs)`
    """

    dones: np.ndarray
    """
    Boolean array indicating episode termination. Shape: (batch_size, ).

    `done[i]` is true iff `next_obs[i]` the last observation of an episode.
    """

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring."""
        super().__post_init__()
        if self.obs.shape != self.next_obs.shape:
            raise ValueError(
                "obs and next_obs must have same shape: "
                f"{self.obs.shape} != {self.next_obs.shape}"
            )
        if self.obs.dtype != self.next_obs.dtype:
            raise ValueError(
                "obs and next_obs must have the same dtype: "
                f"{self.obs.dtype} != {self.next_obs.dtype}"
            )
        if self.dones.shape != (len(self.acts),):
            raise ValueError(
                "dones must be 1D array, one entry for each timestep: "
                f"{self.dones.shape} != ({len(self.acts)},)"
            )
        if self.dones.dtype != bool:
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
        """Performs input validation, including for rews."""
        super().__post_init__()
        _rews_validation(self.rews, self.acts)


def load(path: str) -> Sequence[TrajectoryWithRew]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # TODO(adam): remove backwards compatibility logic eventually (2021?)
    import sys

    try:
        assert "imitation.util.rollout" not in sys.modules
        sys.modules["imitation.util.rollout"] = old_types
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
    finally:
        del sys.modules["imitation.util.rollout"]

    if len(trajectories) > 0:
        if isinstance(trajectories[0], old_types.Trajectory):
            trajectories = [
                TrajectoryWithRew(**traj._asdict()) for traj in trajectories
            ]

    return trajectories


def save(path: str, trajectories: Sequence[TrajectoryWithRew]) -> None:
    """Save a sequence of Trajectories to disk.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(trajectories, f)
    # Ensure atomic write
    os.replace(path + ".tmp", path)
    logging.info("Dumped demonstrations to {}.".format(path))
