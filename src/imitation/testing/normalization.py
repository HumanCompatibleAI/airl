"""Custom environment and assertions for checking that image normalization
works correctly."""

from typing import Optional

import gym
import numpy as np
import stable_baselines3.common.preprocessing as sb3_preproc
import stable_baselines3.common.torch_layers as sb3_torch_layers
import torch as th
from torch import nn

from imitation.rewards import discrim_nets, reward_nets


class NormalizationTestEnv(gym.Env):
    """Environment to test that SB3 "normalisation" is applied exactly once.

    Observations are gradient images that run linearly from some minimum
    intensity value to a maximum intensity value. By default, the minimum is
    slightly above zero (uint8), and the maximum is slightly below 255 (uint8).
    Actions are Discrete(2), but do not have any effect."""

    def __init__(self, *, horizon=3, min_val=1, max_val=254, nchan=3):
        super().__init__()
        self.horizon = horizon
        self.steps = None
        assert isinstance(min_val, int) and isinstance(max_val, int)
        self.min_val = min_val
        self.max_val = max_val
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(nchan, 8, 8), dtype="uint8"
        )
        self.action_space = gym.spaces.Discrete(2)
        obs_shape = self.observation_space.shape
        obs_size = np.prod(obs_shape)
        obs_flat = np.linspace(start=min_val, stop=max_val, num=obs_size, dtype="uint8")
        self._obs = obs_flat.reshape(obs_shape)
        assert self._obs.dtype == np.uint8  # just double-checking…
        self._expected_normalized_obs = self._obs.astype("float32") / 255.0

    def reset(self):
        self.steps = 0
        return self._obs.copy()

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.horizon
        rew = 0.0
        infos = {}
        return self._obs.copy(), rew, done, infos

    def assert_obs_is_normalized(self, tensor):
        # make sure that tensor rescaled by imitation + SB3 is in the right range
        expected_tensor = th.as_tensor(
            self._expected_normalized_obs, device=tensor.device
        )
        is_normalized = th.allclose(
            tensor,
            expected_tensor,
            atol=1e-3,
            rtol=1e-3,
        )
        assert is_normalized, (
            f"input tensor (range [{th.min(tensor)}, {th.max(tensor)}]) is not equal "
            f"to reference tensor (range [{th.min(expected_tensor)}, "
            f"{th.max(expected_tensor)}])"
        )


class NormalizationTestFeatEx(sb3_torch_layers.BaseFeaturesExtractor):
    """Policy feature extractor that checks against ref obs in NormalizationTestEnv."""

    def __init__(self, observation_space, features_dim=11, *, norm_env):
        super().__init__(observation_space, features_dim)
        self._assert_obs_is_normalized = norm_env.assert_obs_is_normalized
        self.assert_calls = 0
        # the "network" here is a simple linear layer on top of the image (it
        # doesn't really matter what it does in our tests---I just chose this
        # so we would have real parameters and gradients)
        in_dim = sb3_preproc.get_flattened_obs_dim(observation_space)
        self.process_net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, features_dim))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        self._assert_obs_is_normalized(obs)
        self.assert_calls += 1
        return self.process_net(obs)


class NormalizationTestDiscriminator(discrim_nets.ActObsMLP):
    """Discriminator for GAIL that checks against ref obs in NormalizationTestEnv."""

    def __init__(self, *args, norm_env, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_obs_is_normalized = norm_env.assert_obs_is_normalized
        self.assert_calls = 0

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        self._assert_obs_is_normalized(obs)
        self.assert_calls += 1
        return super().forward(obs, acts)


class NormalizationTestRewardMLP(reward_nets.BasicRewardMLP):
    """Base reward net for AIRL that checks against ref obs in NormalizationTestEnv."""

    def __init__(
        self,
        *args,
        norm_env,
        use_state=True,
        use_action=True,
        use_next_state=True,
        use_done=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
        )
        self._assert_obs_is_normalized = norm_env.assert_obs_is_normalized
        self.assert_calls = 0

    def forward(
        self,
        state: Optional[th.Tensor] = None,
        action: Optional[th.Tensor] = None,
        next_state: Optional[th.Tensor] = None,
        done: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        if state is not None:
            self._assert_obs_is_normalized(state)
        if next_state is not None:
            self._assert_obs_is_normalized(next_state)
        self.assert_calls += 1
        return super().forward(state, action, next_state, done)
