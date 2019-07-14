"""Constructs deep network reward models."""

from abc import ABC, abstractmethod
import os
import pickle
from typing import Iterable, Optional

import gym
import tensorflow as tf

import imitation.util as util


class RewardNet(ABC):
  """Abstract reward network.

  This class assumes that the caller will set the default TensorFlow Session
  and initialize the network's variables.

  Attributes:
    observation_space: The observation space of `old_obs_ph` and `new_obs_ph`.
    action_space: The action space of `act_ph`.
    old_obs_ph (tf.Tensor): previous observation placeholder.
    act_ph (tf.Tensor): action placeholder.
    new_obs_ph (tf.Tensor): next observation placeholder.
  """

  def __init__(self, observation_space: gym.Space, action_space: gym.Space, *,
               scale: bool = False):
    """Builds a reward network.

    Args:
        observation_space: The observation space.
        action_space: The action space.
        scale: Whether to scale the input.
    """

    self.observation_space = observation_space
    self.action_space = action_space
    self.scale = scale

    inputs = util.build_inputs(observation_space, action_space, scale)
    self.old_obs_ph, self.act_ph, self.new_obs_ph = inputs[:3]
    self.old_obs_inp, self.act_inp, self.new_obs_inp = inputs[3:]

    with tf.variable_scope("theta_network"):
      self._theta_output, self._theta_layers = self.build_theta_network(
          self.old_obs_inp, self.act_inp)

  @property
  @abstractmethod
  def reward_output_train(self):
    """A Tensor holding the training reward associated with each timestep.

    Different concrete subclasses will require different placeholders to be
    filled to calculate this output, but for all subclasses, filling the
    following placeholders will be sufficient:

    ```
    self.old_obs_ph
    self.act_ph
    self.new_obs_ph
    ```

    Returns:
      _reward_output_train: (tf.Tensor) A (None,) shaped Tensor holding
          the training reward associated with each timestep.
    """
    pass

  @property
  def reward_output_test(self):
    """A Tensor holding the test reward associated with each timestep.

    Note this is the reward we use for transfer learning.

    Different concrete subclasses will require different
    placeholders to be filled to calculate this output, but for all
    subclasses, filling the following placeholders will be sufficient:

    ```
    self.old_obs_ph
    self.act_ph
    self.new_obs_ph
    ```

    Returns:
      _reward_output_test: (tf.Tensor) A (None,) shaped Tensor holding
        the test reward associated with each timestep.
    """
    return self._theta_output

  @abstractmethod
  def build_theta_network(self, obs_input, act_input):
    """Builds the test reward network.

    The output of the network is the same as the reward used for transfer
    learning, and is the Tensor returned by `self.reward_output_test()`.

    Args:
      obs_input (tf.Tensor): The observation input. Its shape is
          `((None,) + self.env.observation_space.shape)`.
      act_input (tf.Tensor): The action input. Its shape is
          `((None,) + self.env.action_space.shape)`. The None dimension is
          expected to be the same as None dimension from `obs_input`.

    Returns:
      theta_output (tf.Tensor): A reward prediction for each of the
          inputs. The shape is `(None,)`.
    """
    pass

  def build_summaries(self):
    tf.summary.histogram("train_reward", self.reward_output_train)
    tf.summary.histogram("test_reward", self.reward_output_test)

  @classmethod
  @abstractmethod
  def load(cls, path):
    """Load saved reward network from file."""
    pass

  @abstractmethod
  def save(self, path):
    """Save reward network to file."""
    pass


class RewardNetShaped(RewardNet):
  """Abstract reward network with a phi network to shape training reward.

  This RewardNet formulation matches Equation (4) in the AIRL paper.
  Note that the experiments in Table 2 of the same paper showed shaped
  training rewards to be inferior to an unshaped training rewards in
  a Pendulum environment imitation learning task (and maybe HalfCheetah).
  (See original implementation of Pendulum experiment's reward function at
  https://github.com/justinjfu/inverse_rl/blob/master/inverse_rl/models/imitation_learning.py#L374)

  To make a concrete subclass, implement `build_phi_network()` and
  `build_theta_network()`.
  """

  def __init__(self, observation_space: gym.Space, action_space: gym.Space, *,
               scale: bool = False, discount_factor: float = 0.99):
    super().__init__(observation_space, action_space, scale=scale)
    self._discount_factor = discount_factor

    with tf.variable_scope("phi_network"):
      res = self.build_phi_network(self.old_obs_inp, self.new_obs_inp)
      self._old_shaping_output, self._new_shaping_output = res

    with tf.variable_scope("f_network"):
      self._shaped_reward_output = (
        self._theta_output
        + self._discount_factor * self._new_shaping_output
        - self._old_shaping_output)

  @property
  def reward_output_train(self):
    """A Tensor holding the (shaped) training reward of each timestep.

    Requires the following placeholders to be filled:

    ```
    self.old_obs_ph
    self.act_ph
    self.new_obs_ph
    ```

    Returns:
      _reward_output_train: (tf.Tensor) A (None,) shaped Tensor holding
          the training reward associated with each timestep.
    """
    return self._shaped_reward_output

  @abstractmethod
  def build_phi_network(self, old_obs_input, new_obs_input):
    """Build the reward shaping network (disentangles dynamics from reward).

    XXX: We could potentially make it easier on the subclasser by requiring
    only one input. ie build_phi_network(obs_input). Later in
    _build_f_network, I could stack Tensors of old and new observations,
    pass them simulatenously through the network, and then unstack the
    outputs. Another way to do this would be to pass in a single
    rank 3 obs_input with shape `(2, None) + self.env.observation_space`.

    Args:
      old_obs_input (tf.Tensor): The old observations (corresponding to the
          state at which the current action is made). The shape of this
          Tensor should be `(None,) + self.env.observation_space.shape`.
      new_obs_input (tf.Tensor): The new observations (corresponding to
          the state that we transition to after this state-action pair.

    Returns:
      old_shaping_output (tf.Tensor): A reward shaping prediction for
          each of the old observation inputs. Has shape `(None,)`.
      new_shaping_output (tf.Tensor): A reward shaping prediction for
          each of the new observation inputs. Has shape `(None,)`
    """
    pass

  def build_summaries(self):
    super().build_summaries()
    tf.summary.histogram("shaping_old", self._old_shaping_output)
    tf.summary.histogram("shaping_new", self._new_shaping_output)


def build_basic_theta_network(hid_sizes: Optional[Iterable[int]],
                              old_obs_input: Optional[tf.Tensor],
                              new_obs_input: Optional[tf.Tensor],
                              act_input: Optional[tf.Tensor],
                              **kwargs: dict):
  """Builds a reward network depending on specified observations and actions.

  All specified inputs will be preprocessed and then concatenated. If all
  inputs are specified, then it will be a :math:`R(o,a,o')` network.
  Conversely, if `new_obs_input` and `act_input` are both set to `None`, it
  will depend just on the current observation: :math:`R(o)`.

  Arguments:
    hid_sizes: Number of units at each hidden layer. Default is [], i.e. linear.
    old_obs_input: Previous observation.
    new_obs_input: Next observation.
    act_input: Action.
    kwargs: Passed through to `util.apply_ff`.

  Returns:
    tf.Tensor: Predicted reward.

  Raises:
    ValueError: If all of old_obs_input, new_obs_input and act_input are None.
  """
  if hid_sizes is None:
    hid_sizes = []

  with tf.variable_scope("theta"):
    inputs = [old_obs_input, act_input, new_obs_input]
    inputs = [x for x in inputs if x is not None]
    if len(inputs) == 0:
      raise ValueError("Must specify at least one input")

    inputs = [tf.layers.flatten(x) for x in inputs]
    inputs = tf.concat(inputs, axis=1)
    theta_output, theta_layers = util.apply_ff(inputs, hid_sizes=hid_sizes,
                                               **kwargs)

    return theta_output, theta_layers


class BasicRewardNet(RewardNet):
  """An unshaped reward net with simple, default settings.

  Intended to match the reward network trained for
  the original AIRL pendulum experiments. Right now it has a linear function
  approximator for the theta network, not sure if this is what I want.
  """

  def __init__(self, observation_space: gym.Space, action_space: gym.Space, *,
               scale: bool = False, state_only: bool = False,
               theta_units: Optional[Iterable[int]] = None,
               theta_kwargs: Optional[dict] = None):
    """Builds a simple reward network.

    Args:
      observation_space: The observation space.
      action_space: The action space.
      state_only: If True, then ignore the action when predicting
          and training the reward network theta.
      theta_units: Number of hidden units at each layer of the feedforward
          reward network theta.
      theta_kwargs: Arguments passed to `build_basic_theta_network`.
    """
    self.state_only = state_only
    self.theta_units = theta_units
    self.theta_kwargs = theta_kwargs or {}
    super().__init__(observation_space, action_space, scale=scale)
    # TODO(adam): this is super hacky -- use Sonnet?
    checkpoints = {f'theta_{i}': self._theta_layers[i]
                   for i in range(len(self._theta_layers))}
    self.checkpoint = tf.train.Checkpoint(**checkpoints)

  def build_theta_network(self, obs_input, act_input):
    act_or_none = None if self.state_only else act_input
    return build_basic_theta_network(self.theta_units,
                                     old_obs_input=obs_input,
                                     act_input=act_or_none,
                                     new_obs_input=None,
                                     **self.theta_kwargs)

  @property
  def reward_output_train(self):
    """Training reward is the same as the test reward, since no shaping."""
    return self.reward_output_test

  @classmethod
  def load(cls, path):
    with open(os.path.join(path, 'args'), 'rb') as f:
      params = pickle.load(f)

    obj = cls(**params)
    restore = obj.checkpoint.restore(os.path.join(path, 'weights'))
    # TODO(gleave): assert_consumed
    # restore.assert_consumed().run_restore_ops()
    restore.run_restore_ops()

    return obj

  def save(self, path):
    # TODO(adam): more general than this?
    params = {'observation_space': self.observation_space,
              'action_space': self.action_space,
              'scale': self.scale,
              'state_only': self.state_only,
              'theta_units': self.theta_units,
              'theta_kwargs': self.theta_kwargs}

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'args'), 'wb') as f:
      pickle.dump(params, f)
    self.checkpoint.write(file_prefix=os.path.join(path, 'weights'))


def build_basic_phi_network(hid_sizes: Optional[Iterable[int]],
                            old_obs_input: tf.Tensor,
                            new_obs_input: tf.Tensor,
                            **kwargs: dict):
  """Builds a potential network depending on specified observation.

  Arguments:
    hid_sizes: Number of units at each hidden layer. Default is (32, 32).
    old_obs_input: Previous observation.
    new_obs_input: Next observation.
    kwargs: Passed through to `util.apply_ff`.

  Returns:
    Tuple[tf.Tensor, tf.Tensor]: potential for the old and new observations.
  """
  if hid_sizes is None:
    hid_sizes = (32, 32)

  with tf.variable_scope("phi", reuse=tf.AUTO_REUSE):
    old_o = tf.layers.flatten(old_obs_input)
    new_o = tf.layers.flatten(new_obs_input)

    # Weight share, just with different inputs old_o and new_o
    old_shaping_output = util.apply_ff(old_o, hid_sizes=hid_sizes,
                                       name="old_shaping_output", **kwargs)
    new_shaping_output = util.apply_ff(new_o, hid_sizes=hid_sizes,
                                       name="new_shaping_output", **kwargs)

  return old_shaping_output, new_shaping_output


class BasicShapedRewardNet(RewardNetShaped):
  """A shaped reward network with simple, default settings.

  With default parameters this RewardNet has two hidden layers [32, 32]
  for the reward shaping phi network, and a linear function approximator
  for the theta network. These settings match the network architectures for
  continuous control experiments described in Appendix D.1 of the
  AIRL paper.

  This network flattens inputs. So it isn't suitable for training on
  pixel observations.
  """

  def __init__(self, observation_space: gym.Space, action_space: gym.Space, *,
               scale: bool = False, state_only: bool = False,
               discount_factor: float = 0.99,
               theta_units: Optional[Iterable[int]] = None,
               theta_kwargs: Optional[dict] = None,
               phi_units: Optional[Iterable[int]] = None,
               phi_kwargs: Optional[dict] = None):
    """Builds a simple shaped reward network.

    Args:
      observation_space: The observation space.
      action_space: The action space.
      discount_factor: A number in the range [0, 1].
      state_only: If True, then ignore the action when predicting and training
          the reward network theta.
      theta_units: Number of hidden units at each layer of the feedforward
          reward network theta.
      theta_kwargs: Arguments passed to `build_basic_theta_network`.
      phi_units: Number of hidden units at each layer of the feedforward
          potential network phi.
      phi_kwargs: Arguments passed to `build_basic_phi_network`.
    """
    self.state_only = state_only
    self.theta_units = theta_units
    self.phi_units = phi_units
    self.theta_kwargs = theta_kwargs or {}
    self.phi_kwargs = phi_kwargs or {}
    super().__init__(observation_space, action_space, scale=scale,
                     discount_factor=discount_factor)

  def build_theta_network(self, obs_input, act_input):
    act_or_none = None if self.state_only else act_input
    return build_basic_theta_network(self.theta_units,
                                     old_obs_input=obs_input,
                                     act_input=act_or_none,
                                     new_obs_input=None,
                                     **self.theta_kwargs)

  def build_phi_network(self, old_obs_input, new_obs_input):
    return build_basic_phi_network(self.phi_units, old_obs_input,
                                   new_obs_input, **self.phi_kwargs)
