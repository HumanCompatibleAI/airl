"""Behavioural Cloning (BC). Trains policy by applying supervised learning to a
fixed dataset of (observation, action) pairs generated by some expert
demonstrator."""

from typing import Type

from stable_baselines.common.dataset import Dataset
from stable_baselines.common.policies import ActorCriticPolicy
import tensorflow as tf
from tqdm.autonotebook import tqdm, trange

from imitation.policies.base import FeedForward32Policy
from imitation.util import rollout


class BCTrainer:
  """Simple behavioural cloning (BC).

  Recovers only a policy.

  Args:
    env (gym.Env): environment to train on.
    expert_rollouts: A tuple of four arrays from expert rollouts,
        `obs`, `act`, `next_obs`, `reward`.
    policy_class (BasePolicy): used to instantiate imitation policy.
    batch_size (int): batch size used for training.
    """

  def __init__(self,
               env,
               *,
               expert_demos: rollout.Transitions,
               policy_class: Type[ActorCriticPolicy] = FeedForward32Policy,
               batch_size: int = 32):
    self.env = env
    self.policy_class = policy_class
    self.batch_size = batch_size
    self.expert_dataset = Dataset(
        {
            "obs": expert_demos.obs,
            "act": expert_demos.acts,
        }, shuffle=True)
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      self._build_tf_graph()
      self.sess.run(tf.global_variables_initializer())

  def train(self, *, n_epochs=100):
    """Train with supervised learning for some number of epochs.

    Here an 'epoch' is just a complete pass through the expert transition
    dataset.

    Args:
      n_epochs (int): number of complete passes made through dataset.
    """
    epoch_iter = trange(n_epochs, desc='BC epoch')
    for epoch_num in epoch_iter:
      total_batches = self.expert_dataset.n_samples // self.batch_size
      batch_iter = self.expert_dataset.iterate_once(self.batch_size)
      tq_iter = tqdm(
          batch_iter, total=total_batches, desc='pol step', leave=False)
      loss_ewma = None
      for batch_dict in tq_iter:
        feed_dict = {
            self._true_acts_ph: batch_dict['act'],
            self.policy.obs_ph: batch_dict['obs'],
        }
        _, loss = self.sess.run(
            [self._train_op, self._log_loss], feed_dict=feed_dict)
        tq_iter.set_postfix(loss='% 3.4f' % loss)
        if loss_ewma is None:
          loss_ewma = loss
        else:
          loss_ewma = 0.9 * loss_ewma + 0.1 * loss
      epoch_iter.set_postfix(loss_ewma='% 3.4f' % loss_ewma)

  def test_policy(self, *, n_episodes=10):
    """Test current imitation policy on environment & give some rollout
    stats.

    Args:
      n_episodes (int): number of rolled-out episodes.

    Returns:
      dict: rollout statistics collected by
        `imitation.utils.rollout.rollout_stats()`.
    """
    reward_stats = rollout.rollout_stats(
        self.policy, self.env, sample_until=rollout.min_episodes(n_episodes))
    return reward_stats

  def _build_tf_graph(self):
    with tf.name_scope('bc_supervised_loss'):
      self.policy = self.policy_class(
          self.sess,
          self.env.observation_space,
          self.env.action_space,
          n_batch=None,
          n_env=1,
          n_steps=1000)  # pytype: disable=not-instantiable
      self._true_acts_ph = self.policy.pdtype.sample_placeholder(
          [None], name='ref_acts_ph')
      self._log_loss = tf.reduce_mean(
          self.policy.proba_distribution.neglogp(self._true_acts_ph))
      opt = tf.train.AdamOptimizer()
      self._train_op = opt.minimize(self._log_loss)
