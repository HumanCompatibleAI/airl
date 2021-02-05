import itertools

import gym
import numpy as np
import pytest
import torch.utils.data as th_data
from numpy import testing as npt

from imitation.contrib import data_loader_wrappers as dwrappers
from imitation.data import types
from imitation.util import util


class Transform1(dwrappers.Transform):
    def down_acts(self, acts):
        return acts - 1

    def up_acts(self, acts):
        return acts + 1

    def up_obs(self, obs):
        return obs + 1

    def up_rews(self, rews):
        return rews + 1


class GymWrapperWithMixinLikeTransform1(
    gym.Wrapper, dwrappers.ApplyDataLoaderWrapperMixin
):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def apply_data_loader_wrapper(
        self,
        data_loader,
    ):
        transform = Transform1()
        return dwrappers.DataLoaderWrapperFromTransform(data_loader, transform)


@pytest.fixture(params=[1, 30, 127])
def dummy_batch_size(request):
    return request.param


@pytest.fixture
def dummy_data(dummy_batch_size):
    return dict(
        obs=np.zeros([dummy_batch_size, 1]),
        acts=np.zeros([dummy_batch_size, 3]),
        next_obs=np.zeros([dummy_batch_size, 1]),
        infos=np.array([{} for _ in range(dummy_batch_size)]),
        rews=np.zeros([dummy_batch_size]),
    )


class DummyDataLoader:
    def __init__(self, data: dict, use_pytorch: bool):
        self.data = data
        self.use_pytorch = use_pytorch

    def __iter__(self):
        if self.use_pytorch:
            dones = np.zeros(len(self.data["obs"]), dtype=bool)
            dataset = types.TransitionsWithRew(**self.data, dones=dones)
            data_loader = th_data.DataLoader(
                dataset=dataset,
                batch_size=len(self.data["obs"]),
                collate_fn=types.transitions_collate_fn,
            )
            yield from util.endless_iter(data_loader)
        else:
            yield from itertools.repeat(self.data)


@pytest.fixture(params=[True, False])
def dummy_use_pytorch_backend(request):
    return request.param


@pytest.fixture
def dummy_data_loader(dummy_data, dummy_use_pytorch_backend):
    dl = DummyDataLoader(dummy_data, dummy_use_pytorch_backend)

    if dummy_use_pytorch_backend:
        # Expecting Numpy array, not Tensors, on the other side,
        # so let's preemptively convert.
        dl = dwrappers.TensorToNumpyDataLoaderWrapper(dl)
    return dl


class DummyEnv(gym.Env):
    def __init__(self, data: dict):
        self.data = data
        self.observation_space = data["obs"][0].shape
        self.action_space = data["acts"][0].shape
        self.saved_actions = []

    def step(self, action):
        self.saved_actions.append(action)
        ob = self.data["obs"][0]
        rew = self.data["rews"][0]
        info = self.data["infos"][0]
        done = True
        return ob, rew, done, info


@pytest.fixture
def dummy_env(dummy_data):
    return DummyEnv(dummy_data)


def _check_transform1_data_loader(dummy_data, data_loader, n_wrappers):
    for i, data in zip(range(3), data_loader):
        assert np.all(dummy_data["infos"] == dummy_data["infos"])
        for key in ["obs", "acts", "next_obs", "rews"]:
            npt.assert_array_almost_equal(dummy_data[key] + n_wrappers, data[key])


def test_transform1_data_loader_wrappers(dummy_data, dummy_data_loader, n_wrappers=1):
    dl = dummy_data_loader

    for _ in range(n_wrappers):
        transform = Transform1()
        dl = dwrappers.DataLoaderWrapperFromTransform(dl, transform)

    _check_transform1_data_loader(dummy_data, dl, n_wrappers)


def test_transform1_env_wrappers_and_auto_data_loader_wrapper(
    dummy_data,
    dummy_env,
    dummy_data_loader,
    n_wrappers=1,
):
    dl = dummy_data_loader
    env = dummy_env
    transform = Transform1()
    for _ in range(n_wrappers):
        env = dwrappers.GymWrapperFromTransform(env, transform)

    # Check auto-wrapped dataloader.
    wrapped_dl = dwrappers.wrap_dataset_with_env_wrappers(dl, env)
    _check_transform1_data_loader(dummy_data, wrapped_dl, n_wrappers)

    # Check env step.
    for i, data in zip(range(3), dl):
        act = data["acts"][0]
        ob, rew, done, info = env.step(act)

        assert np.all(info == dummy_data["infos"][0])
        npt.assert_array_almost_equal(dummy_data["obs"][0] + n_wrappers, ob)
        npt.assert_array_almost_equal(dummy_data["rews"][0] + n_wrappers, rew)
        assert len(dummy_env.saved_actions) == i + 1
        npt.assert_array_almost_equal(dummy_env.saved_actions[-1], act - n_wrappers)

    assert dummy_env.saved_actions
