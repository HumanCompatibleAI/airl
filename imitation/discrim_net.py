from abc import abstractmethod
import os
import pickle
from typing import Iterable, Optional

import gym
import tensorflow as tf

from imitation import reward_net, util
from imitation.util import serialize


class DiscrimNet(serialize.Serializable):
  """Abstract base class for discriminator, used in multiple IRL methods."""

  def __init__(self):
    self._disc_loss = self.build_disc_loss()
    self._policy_train_reward = self.build_policy_train_reward()
    self._policy_test_reward = self.build_policy_test_reward()

  @property
  def disc_loss(self):
    return self._disc_loss

  @property
  def policy_train_reward(self):
    return self._policy_train_reward

  @property
  def policy_test_reward(self):
    return self._policy_test_reward

  @abstractmethod
  def build_policy_train_reward(self):
    pass

  @abstractmethod
  def build_policy_test_reward(self):
    pass

  @abstractmethod
  def build_disc_loss(self):
    # Holds the label of every state-action pair that the discriminator
    # is being trained on. Use 0.0 for expert policy. Use 1.0 for generated
    # policy.
    self._labels_ph = tf.placeholder(shape=(None,), dtype=tf.int32,
                                     name="discrim_labels")

    # This placeholder holds the generator-policy log action probabilities,
    # $\log \pi(a \mid s)$, of each state-action pair. This includes both
    # actions taken by the generator *and* those by the expert (we can
    # ask our policy for action probabilities even if we don't take them).
    # TODO(adam): this is only used by AIRL; sad we have to always include it
    self._log_policy_act_prob_ph = tf.placeholder(
        shape=(None,), dtype=tf.float32, name="log_ro_act_prob_ph")

  @abstractmethod
  def build_summaries(self):
    pass

  @property
  @abstractmethod
  def old_obs_ph(self):
    """The old observation placeholder."""
    pass

  @property
  @abstractmethod
  def act_ph(self):
    """The action placeholder."""
    pass

  @property
  @abstractmethod
  def new_obs_ph(self):
    """The new observation placeholder."""
    pass

  @property
  def labels_ph(self):
    """The expert (0.0) or generated (1.0) labels placeholder."""
    return self._labels_ph

  @property
  def log_policy_act_prob_ph(self):
    """The log-probability of policy actions placeholder."""
    return self._log_policy_act_prob_ph


class DiscrimNetAIRL(DiscrimNet):
  r"""The AIRL discriminator for a given RewardNet.

  The AIRL discriminator is of the form
  .. math:: D_{\theta} = \frac{\exp(f_{\theta}(s,a)}{\exp(f_{\theta}(s,a) + \pi(a \mid s)}

  where :math:`f_{\theta}` is `self.reward_net`.
  """  # noqa: E501

  def __init__(self,
               reward_net: reward_net.RewardNet,
               entropy_weight: float = 1.0):
    """Builds a DiscrimNetAIRL.

    Args:
        reward_net: A RewardNet, used as $f_{\theta}$ in the discriminator.
        entropy_weight: The coefficient for the entropy regularization term.
            To match the AIRL derivation, it should be 1.0.
            However, empirically a lower value sometimes work better.
    """
    self.reward_net = reward_net
    self.entropy_weight = entropy_weight
    super().__init__()
    tf.logging.info("Using AIRL")

  @property
  def old_obs_ph(self):
    return self.reward_net.old_obs_ph

  @property
  def act_ph(self):
    return self.reward_net.act_ph

  @property
  def new_obs_ph(self):
    return self.reward_net.new_obs_ph

  def build_summaries(self):
    self.reward_net.build_summaries()

  def build_disc_loss(self):
    super().build_disc_loss()

    # The AIRL discriminator is trained with the cross-entropy loss between
    # expert demonstrations and generated samples.

    # Construct discriminator logits: $f_{\theta}(s,a)$, predicted rewards,
    # and $\log \pi(a \mid s)$, generator-policy log action probabilities.
    self._presoftmax_disc_logits = tf.stack(
        [self.reward_net.reward_output_train, self.log_policy_act_prob_ph],
        axis=1, name="presoftmax_discriminator_logits")  # (None, 2)

    # Construct discriminator loss.
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.labels_ph,
        logits=self._presoftmax_disc_logits,
        name="discrim_loss",
    )

  def build_policy_test_reward(self):
    super().build_policy_test_reward()
    return self.reward_net.reward_output_test

  def build_policy_train_reward(self):
    """
    Sets self._policy_train_reward_fn, the reward function to use when
    running a policy optimizer (e.g. PPO).
    """
    # Construct generator reward:
    # \[\hat{r}(s,a) = \log(D_{\theta}(s,a)) - \log(1 - D_{\theta}(s,a)).\]
    # This simplifies to:
    # \[\hat{r}(s,a) = f_{\theta}(s,a) - \log \pi(a \mid s).\]
    # This is just an entropy-regularized objective.
    self._log_D = tf.nn.log_softmax(self.reward_net.reward_output_train)
    self._log_D_compl = tf.nn.log_softmax(self.log_policy_act_prob_ph)
    # Note self._log_D_compl is effectively an entropy term.
    return self._log_D - self.entropy_weight * self._log_D_compl

  def save(self, directory):
    params = {
        "entropy_weight": self.entropy_weight,
        "reward_net_cls": type(self.reward_net),
    }
    with open(os.path.join(directory, "args"), "wb") as f:
      pickle.dump(params, f)
    return self.reward_net.save(os.path.join(directory, "reward_net"))

  @classmethod
  def load(cls, directory):
    with open(os.path.join(directory, "args"), "rb") as f:
      params = pickle.load(f)
    reward_net_cls = params.pop("reward_net_cls")
    reward_net = reward_net_cls.load(os.path.join(directory, "reward_net"))
    return cls(reward_net=reward_net, **params)


class DiscrimNetGAIL(DiscrimNet, serialize.LayersSerializable):
  """The discriminator to use for GAIL."""

  def __init__(self,
               observation_space: gym.Space,
               action_space: gym.Space,
               hid_sizes: Optional[Iterable[int]] = None,
               scale: bool = False):
    args = locals()
    inputs = util.build_inputs(observation_space, action_space, scale=scale)
    self._old_obs_ph, self._act_ph, self._new_obs_ph = inputs[:3]
    self.old_obs_inp, self.act_inp, self.new_obs_inp = inputs[3:]

    self.hid_sizes = hid_sizes
    with tf.variable_scope("discrim_network"):
      discrim_mlp, self._discrim_logits = self.build_discrm_network(
          self.old_obs_inp, self.act_inp)

    DiscrimNet.__init__(self)
    serialize.LayersSerializable.__init__(**args, layers=discrim_mlp)

    tf.logging.info("using GAIL")

  @property
  def old_obs_ph(self):
    return self._old_obs_ph

  @property
  def act_ph(self):
    return self._act_ph

  @property
  def new_obs_ph(self):
    return self._new_obs_ph

  def build_discrm_network(self, obs_input, act_input):
    inputs = tf.concat([
        tf.layers.flatten(obs_input),
        tf.layers.flatten(act_input)], axis=1)

    hid_sizes = self.hid_sizes
    if hid_sizes is None:
      hid_sizes = (32, 32)
    discrim_mlp = util.build_mlp(hid_sizes=hid_sizes)
    discrim_logits = util.sequential(inputs, discrim_mlp)

    return discrim_mlp, discrim_logits

  def build_policy_train_reward(self):
    super().build_policy_train_reward()
    train_reward = -tf.log_sigmoid(self._discrim_logits)

    return train_reward

  def build_policy_test_reward(self):
    super().build_policy_test_reward()
    return self.build_policy_train_reward()

  def build_disc_loss(self):
    super().build_disc_loss()

    disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._discrim_logits,
        labels=tf.cast(self.labels_ph, tf.float32),
    )

    return disc_loss

  def build_summaries(self):
    super().build_summaries()
    tf.summary.histogram("discrim_logits", self._discrim_logits)
