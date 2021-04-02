"""Tests of `imitation.data.{dataset,types}`."""


import contextlib
import dataclasses
import os
import pathlib
import pickle
from typing import Any, Callable

import gym
import numpy as np
import pytest

from imitation.data import types

SPACES = [
    gym.spaces.Discrete(3),
    gym.spaces.MultiDiscrete((3, 4)),
    gym.spaces.Box(-1, 1, shape=(1,)),
    gym.spaces.Box(-1, 1, shape=(2,)),
    gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
]
OBS_SPACES = SPACES
ACT_SPACES = SPACES
LENGTHS = [0, 1, 2, 10]


def _check_1d_shape(fn: Callable[[np.ndarray], Any], length: float, expected_msg: str):
    for shape in [(), (length, 1), (length, 2), (length - 1,), (length + 1,)]:
        with pytest.raises(ValueError, match=expected_msg):
            fn(np.zeros(shape))


@pytest.fixture
def trajectory(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> types.Trajectory:
    """Fixture to generate trajectory of length `length` iid sampled from spaces."""
    if length == 0:
        pytest.skip()
    obs = np.array([obs_space.sample() for _ in range(length + 1)])
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = np.array([{i: i} for i in range(length)])
    return types.Trajectory(obs=obs, acts=acts, infos=infos)


@pytest.fixture
def trajectory_rew(trajectory: types.Trajectory) -> types.TrajectoryWithRew:
    """Like `trajectory` but with reward randomly sampled from a Gaussian."""
    rews = np.random.randn(len(trajectory))
    return types.TrajectoryWithRew(**dataclasses.asdict(trajectory), rews=rews)


@pytest.fixture
def transitions_min(
    obs_space: gym.Space, act_space: gym.Space, length: int
) -> types.TransitionsMinimal:
    obs = np.array([obs_space.sample() for _ in range(length)])
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = np.array([{i: i} for i in range(length)])
    return types.TransitionsMinimal(obs=obs, acts=acts, infos=infos)


@pytest.fixture
def transitions(
    transitions_min: types.TransitionsMinimal, obs_space: gym.Space, length: int
) -> types.Transitions:
    """Fixture to generate transitions of length `length` iid sampled from spaces."""
    next_obs = np.array([obs_space.sample() for _ in range(length)])
    dones = np.zeros(length, dtype=bool)
    return types.Transitions(
        **dataclasses.asdict(transitions_min), next_obs=next_obs, dones=dones
    )


@pytest.fixture
def transitions_rew(
    transitions: types.Transitions, length: int
) -> types.TransitionsWithRew:
    """Like `transitions` but with reward randomly sampled from a Gaussian."""
    rews = np.random.randn(length)
    return types.TransitionsWithRew(**dataclasses.asdict(transitions), rews=rews)


def _check_transitions_get_item(trans, key):
    """Check trans[key] by manually indexing/slicing into every `trans` field."""
    item = trans[key]
    for field in dataclasses.fields(trans):
        if isinstance(item, dict):
            observed = item[field.name]  # pytype: disable=unsupported-operands
        else:
            observed = getattr(item, field.name)

        expected = getattr(trans, field.name)[key]
        if isinstance(expected, np.ndarray):
            assert observed.dtype == expected.dtype
        np.testing.assert_array_equal(observed, expected)


@contextlib.contextmanager
def pushd(dir_path):
    """Change directory temporarily inside context."""
    orig_dir = os.getcwd()
    try:
        os.chdir(dir_path)
        yield
    finally:
        os.chdir(orig_dir)


@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
@pytest.mark.parametrize("length", LENGTHS)
class TestData:
    """Tests of imitation.util.data.

    Grouped in a class since parametrized over common set of spaces.
    """

    def test_valid_trajectories(
        self,
        trajectory: types.Trajectory,
        trajectory_rew: types.TrajectoryWithRew,
        length: int,
    ) -> None:
        """Checks trajectories can be created for a variety of lengths and spaces."""
        trajs = [trajectory, trajectory_rew]
        trajs += [dataclasses.replace(traj, infos=None) for traj in trajs]
        for traj in trajs:
            assert len(traj) == length

    @pytest.mark.parametrize("use_chdir", [False, True])
    def test_save_trajectories(self, trajectory_rew, use_chdir, tmpdir):
        """Check that trajectories are properly saved."""
        if use_chdir:
            # Test no relative path without directory edge-case.
            chdir_context = pushd(tmpdir)
            save_dir = ""
        else:
            chdir_context = contextlib.nullcontext()
            save_dir = tmpdir

        trajs = [trajectory_rew]
        save_path = pathlib.Path(save_dir, "trajs.pkl")

        with chdir_context:
            types.save(str(save_path), trajs)
            with open(save_path, "rb") as f:
                loaded_trajs = pickle.load(f)
            assert len(trajs) == len(loaded_trajs)
            for t1, t2 in zip(trajs, loaded_trajs):
                d1, d2 = dataclasses.asdict(t1), dataclasses.asdict(t2)
                assert d1.keys() == d2.keys()
                for k, v in d1.items():
                    assert np.array_equal(v, d2[k])

    def test_invalid_trajectories(
        self,
        trajectory: types.Trajectory,
        trajectory_rew: types.TrajectoryWithRew,
    ) -> None:
        """Checks input validation catches space and dtype related errors."""
        trajs = [trajectory, trajectory_rew]
        for traj in trajs:
            with pytest.raises(
                ValueError, match=r"expected one more observations than actions.*"
            ):
                dataclasses.replace(traj, obs=traj.obs[:-1])
            with pytest.raises(
                ValueError, match=r"expected one more observations than actions.*"
            ):
                dataclasses.replace(traj, acts=traj.acts[:-1])

            with pytest.raises(
                ValueError,
                match=r"infos when present must be present for each action.*",
            ):
                dataclasses.replace(traj, infos=traj.infos[:-1])
            with pytest.raises(
                ValueError,
                match=r"infos when present must be present for each action.*",
            ):
                dataclasses.replace(traj, obs=traj.obs[:-1], acts=traj.acts[:-1])

        _check_1d_shape(
            fn=lambda rews: dataclasses.replace(trajectory_rew, rews=rews),
            length=len(trajectory_rew),
            expected_msg=r"rewards must be 1D array.*",
        )

        with pytest.raises(ValueError, match=r"rewards dtype.* not a float"):
            dataclasses.replace(
                trajectory_rew, rews=np.zeros(len(trajectory_rew), dtype=int)
            )

    def test_valid_transitions(
        self,
        transitions_min: types.TransitionsMinimal,
        transitions: types.Transitions,
        transitions_rew: types.TransitionsWithRew,
        length: int,
        n_checks: int = 20,
    ) -> None:
        """Checks initialization, indexing, and slicing sanity."""
        for trans in [transitions_min, transitions, transitions_rew]:
            assert len(trans) == length

            for _ in range(n_checks):
                # Indexing checks, which require at least one element.
                if length != 0:
                    index = np.random.randint(length)
                    assert isinstance(trans[index], dict)
                    _check_transitions_get_item(trans, index)

                # Slicing checks.
                start = np.random.randint(-2, length)
                stop = np.random.randint(0, length + 2)
                step = np.random.randint(-2, 4)
                if step == 0:  # Illegal. Quick fix that biases tests to ordinary step.
                    step = 1
                s = slice(start, stop, step)
                assert type(trans[s]) is type(trans)
                _check_transitions_get_item(trans, s)

    def test_invalid_transitions(
        self,
        transitions_min: types.Transitions,
        transitions: types.Transitions,
        transitions_rew: types.TransitionsWithRew,
        length: int,
    ) -> None:
        """Checks input validation catches space and dtype related errors."""
        if length == 0:
            pytest.skip()

        for trans in [transitions_min, transitions, transitions_rew]:
            with pytest.raises(
                ValueError, match=r"obs and acts must have same number of timesteps:.*"
            ):
                dataclasses.replace(trans, acts=trans.acts[:-1])
            with pytest.raises(
                ValueError, match=r"obs and infos must have same number of timesteps:.*"
            ):
                dataclasses.replace(trans, infos=[{}] * (length - 1))

        for trans in [transitions, transitions_rew]:
            with pytest.raises(
                ValueError, match=r"obs and next_obs must have same shape:.*"
            ):
                dataclasses.replace(trans, next_obs=np.zeros((len(trans), 4, 2)))

            with pytest.raises(
                ValueError, match=r"obs and next_obs must have the same dtype:.*"
            ):
                dataclasses.replace(
                    trans, next_obs=np.zeros_like(trans.next_obs, dtype=bool)
                ),

            _check_1d_shape(
                fn=lambda bogus_dones: dataclasses.replace(trans, dones=bogus_dones),
                length=len(trans),
                expected_msg=r"dones must be 1D array.*",
            )

            with pytest.raises(ValueError, match=r"dones must be boolean"):
                dataclasses.replace(trans, dones=np.zeros(len(trans), dtype=int))

        _check_1d_shape(
            fn=lambda bogus_rews: dataclasses.replace(trans, rews=bogus_rews),
            length=len(transitions_rew),
            expected_msg=r"rewards must be 1D array.*",
        )

        with pytest.raises(ValueError, match=r"rewards dtype.* not a float"):
            dataclasses.replace(
                transitions_rew, rews=np.zeros(len(transitions_rew), dtype=int)
            )


def test_zero_length_fails():
    """Check zero-length trajectory and transitions fail."""
    empty = np.array([])
    with pytest.raises(ValueError, match=r"Degenerate trajectory.*"):
        types.Trajectory(obs=np.array([42]), acts=empty, infos=None)
