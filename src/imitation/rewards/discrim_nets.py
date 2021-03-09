import abc
import logging
from typing import Optional

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common import preprocessing
from torch import nn

from imitation.rewards import common as rewards_common
from imitation.rewards import reward_nets
from imitation.util import networks


class DiscrimNet(nn.Module, abc.ABC):
    """Abstract base class for discriminator, used in AIRL and GAIL."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        normalize_images: bool = True,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images

    @abc.abstractmethod
    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.

        Args:
            state: state at time t.
            action: action taken at time t.
            next_state: state at time t+1.
            done: binary episode completion flag after action at time t.
            log_policy_act_prob: log policy of novice taking `action`. This is
                only used for AIRL.

        Returns:
            disc_logits_gen_is_high: discriminator logits for a sigmoid
                activation. A high output indicates a generator-like transition.
        """

    def disc_loss(self, disc_logits_gen_is_high, labels_gen_is_one) -> th.Tensor:
        """Compute discriminator loss.

        Args:
            disc_logits_gen_is_high: discriminator logits, as produced by
                `logits_gen_is_high`.
            labels_gen_is_one: integer labels, with zero for expert and one for
                generator (novice).

        Returns:
            loss: scalar-valued discriminator loss."""
        return F.binary_cross_entropy_with_logits(
            disc_logits_gen_is_high, labels_gen_is_one.float()
        )

    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device

    @abc.abstractmethod
    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Test-time reward for given states/actions."""

    @abc.abstractmethod
    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Train-time reward for given states/actions."""

    def predict_reward_train(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an imitation learning algorithm.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=True, state=state, action=action, next_state=next_state, done=done
        )

    def predict_reward_test(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an expert during transfer learning.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            act: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=False, state=state, action=action, next_state=next_state, done=done
        )

    def _eval_reward(
        self,
        is_train: bool,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> np.ndarray:
        (
            state_th,
            action_th,
            next_state_th,
            done_th,
        ) = rewards_common.disc_rew_preprocess_inputs(
            observation_space=self.observation_space,
            action_space=self.action_space,
            state=state,
            action=action,
            next_state=next_state,
            done=done,
            device=self.device(),
            normalize_images=self.normalize_images,
        )

        with th.no_grad():
            if is_train:
                rew_th = self.reward_train(state_th, action_th, next_state_th, done_th)
            else:
                rew_th = self.reward_test(state_th, action_th, next_state_th, done_th)

        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == (len(state),)

        return rew


class DiscrimNetAIRL(DiscrimNet):
    r"""The AIRL discriminator for a given RewardNet.

    The AIRL discriminator is of the form
    .. math:: D_{\theta}(s,a) = \frac{\exp(f_{\theta}(s,a)}{\exp(f_{\theta}(s,a) + \pi(a \mid s)}

    where :math:`f_{\theta}` is `self.reward_net`.
    """  # noqa: E501

    def __init__(self, reward_net: reward_nets.RewardNet, entropy_weight: float = 1.0):
        """Builds a DiscrimNetAIRL.

        Args:
            reward_net: A RewardNet, used as $f_{\theta}$ in the discriminator.
            entropy_weight: The coefficient for the entropy regularization term.
                To match the AIRL derivation, it should be 1.0.
                However, empirically a lower value sometimes work better.
        """
        super().__init__(
            observation_space=reward_net.observation_space,
            action_space=reward_net.action_space,
        )
        self.reward_net = reward_net
        self.entropy_weight = entropy_weight
        logging.info("Using AIRL")

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: th.Tensor,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.
        """
        reward_output_train = self.reward_net.reward_train(
            state, action, next_state, done
        )
        # In Fu's AIRL paper (https://arxiv.org/pdf/1710.11248.pdf), the
        # discriminator output was given as exp(r_theta(s,a)) /
        # (exp(r_theta(s,a)) - log pi(a|s)), with a high value corresponding to
        # expert and a low value corresponding to generator (the opposite of
        # our convention).
        #
        # Observe that sigmoid(log pi(a|s) - r(s,a)) = exp(log pi(a|s) -
        # r(s,a)) / (1 + exp(log pi(a|s) - r(s,a))). If we multiply through by
        # exp(r(s,a)), we get pi(a|s) / (pi(a|s) + exp(r(s,a))). This is the
        # original AIRL discriminator expression with reversed logits to match
        # our convention of low = expert and high = generator (like GAIL).
        return log_policy_act_prob - reward_output_train

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        rew = self.reward_net.reward_test(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute train reward.

        Computed reward does *not* include an entropy bonus. Instead, the
        entropy bonus should be added directly to PPO, SAC, etc."""
        rew = self.reward_net.reward_train(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew


class ActObsMLP(nn.Module):
    """Simple MLP that takes an action and observation and produces a single
    output."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        in_size = preprocessing.get_flattened_obs_dim(
            observation_space
        ) + preprocessing.get_flattened_obs_dim(action_space)
        self.mlp = networks.build_mlp(
            **{"in_size": in_size, "out_size": 1, **mlp_kwargs}
        )

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        flat_obs = th.flatten(obs, start_dim=1)
        flat_acts = th.flatten(acts, start_dim=1)
        cat_inputs = th.cat((flat_obs, flat_acts), dim=1)
        outputs = self.mlp(cat_inputs)
        return outputs.squeeze(1)


class DiscrimNetGAIL(DiscrimNet):
    """The discriminator to use for GAIL."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        discrim_net: Optional[nn.Module] = None,
        normalize_images: bool = True,
    ):
        """Construct discriminator network.

        Args:
          observation_space: observation space for this environment.
          action_space: action space for this environment:
          discrim_net: a Torch module that takes an observation and action
            tensor as input, then computes the logits for GAIL.
          normalize_images: should uint8 image inputs be rescaled to floats in
            [0,1]?
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            normalize_images=normalize_images,
        )

        if discrim_net is None:
            self.discriminator = ActObsMLP(
                action_space=action_space,
                observation_space=observation_space,
                hid_sizes=(32, 32),
            )
        else:
            self.discriminator = discrim_net

        logging.info("using GAIL")

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.
        """
        logits = self.discriminator(state, action)
        return logits

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        rew = self.reward_train(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.logits_gen_is_high(state, action, next_state, done)
        rew = -F.logsigmoid(logits)
        assert rew.shape == state.shape[:1]
        return rew
