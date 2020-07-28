"""Behavioural Cloning (BC).

Trains policy by applying supervised learning to a fixed dataset of (observation,
action) pairs generated by some expert demonstrator.
"""

from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union

import gym
import torch as th
from stable_baselines3.common import logger, policies, utils
from tqdm.autonotebook import trange

from imitation.data import datasets, types
from imitation.policies import base


def reconstruct_policy(
    policy_path: str, device: Union[th.device, str] = "auto",
) -> policies.BasePolicy:
    """Reconstruct a saved policy.

    Args:
        policy_path: path where `.save_policy()` has been run.
        device: device on which to load the policy.

    Returns:
        policy: policy with reloaded weights.
    """
    policy = th.load(policy_path, map_location=utils.get_device(device))
    assert isinstance(policy, policies.BasePolicy)
    return policy


class ConstantLRSchedule:
    """A callable that returns a constant learning rate."""

    def __init__(self, lr: float = 1e-3):
        """
        Args:
            lr: the constant learning rate that calls to this object will return.
        """
        self.lr = lr

    def __call__(self, _):
        """
        Returns the constant learning rate.
        """
        return self.lr


class BC:
    # TODO(scottemmons): pass BasePolicy into BC directly (rather than passing its
    #  arguments)
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        policy_class: Type[policies.BasePolicy] = base.FeedForward32Policy,
        policy_kwargs: Optional[Mapping[str, Any]] = None,
        expert_data: Union[
            types.TransitionsMinimal, datasets.Dataset[types.TransitionsMinimal], None,
        ] = None,
        batch_size: int = 32,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
    ):
        """Behavioral cloning (BC).

        Recovers a policy via supervised learning on a Dataset of observation-action
        pairs.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            policy_class: used to instantiate imitation policy.
            policy_kwargs: keyword arguments passed to policy's constructor.
            expert_data: If not None, then immediately call
                  `self.set_expert_dataset(expert_data)` during initialization.
            batch_size: batch size used for training.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                  weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
        """
        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError(
                    "Use the parameter l2_weight to specify the weight decay."
                )

        self.action_space = action_space
        self.observation_space = observation_space
        self.policy_class = policy_class
        self.policy_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=ConstantLRSchedule(),
        )
        self.policy_kwargs.update(policy_kwargs or {})

        self.policy = self.policy_class(
            **self.policy_kwargs
        )  # pytype: disable=not-instantiable
        # FIXME(sam): this is to get around SB3 bug that fails to put
        # action_net, value_net, etc. on the same device as mlp_extractor in
        # ActorCriticPolicy. Remove this once SB3 issue #111 is fixed.
        self.policy = self.policy.to(self.policy.device)
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(self.policy.parameters(), **optimizer_kwargs)

        assert batch_size >= 1
        self.batch_size = batch_size
        self.expert_dataset: Optional[datasets.Dataset[types.TransitionsMinimal]] = None
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

        if expert_data is not None:
            self.set_expert_dataset(expert_data)

    def set_expert_dataset(
        self,
        expert_data: Union[
            types.TransitionsMinimal, datasets.Dataset[types.TransitionsMinimal],
        ],
    ) -> None:
        """Replace the current expert dataset with a new one.

        Useful for DAgger and other interactive algorithms.

        Args:
             expert_data: Either a `Dataset[types.TransitionsMinimal]` for which
                 `.size()` is not None, or a instance of `TransitionsMinimal`, which
                 is automatically converted to a shuffled, epoch-order
                 `Dataset[types.TransitionsMinimal]`.
        """
        if isinstance(expert_data, types.Transitions):
            trans = expert_data
            expert_dataset = datasets.TransitionsDictDatasetAdaptor(
                trans, datasets.EpochOrderDictDataset
            )
        else:
            assert isinstance(expert_data, datasets.Dataset)
            expert_dataset = expert_data
        assert expert_dataset.size() is not None
        self.expert_dataset = expert_dataset

    def _calculate_loss(self, obs, acts) -> Tuple[th.Tensor, Dict[str, float]]:
        """
        Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            loss: The supervised learning loss for the behavioral clone to optimize.
            stats_dict: Statistics about the learning process to be logged.

        """
        _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        l2_norms = [th.sum(th.square(w)) for w in self.policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        stats_dict = dict(
            neglogp=neglogp.item(),
            loss=loss.item(),
            entropy=entropy.item(),
            ent_loss=ent_loss.item(),
            prob_true_act=prob_true_act.item(),
            l2_norm=l2_norm.item(),
            l2_loss=l2_loss.item(),
        )

        return loss, stats_dict

    def train(
        self,
        n_epochs: int = 100,
        *,
        on_epoch_end: Callable[[dict], None] = None,
        log_interval: int = 100,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert transition
        dataset.

        Args:
          n_epochs: number of complete passes made through dataset.
          on_epoch_end: optional callback to run at
            the end of each epoch. Will receive all locals from this function as
            dictionary argument (!!).
          log_interval: log stats after every log_interval batches
        """
        assert self.batch_size >= 1
        samples_so_far = 0
        batch_num = 0
        for epoch_num in trange(n_epochs, desc="BC epoch"):
            while samples_so_far < (epoch_num + 1) * self.expert_dataset.size():
                batch_num += 1
                trans = self.expert_dataset.sample(self.batch_size)
                assert len(trans) == self.batch_size
                samples_so_far += self.batch_size

                obs_tensor = th.as_tensor(trans.obs).to(self.policy.device)
                acts_tensor = th.as_tensor(trans.acts).to(self.policy.device)
                loss, stats_dict = self._calculate_loss(obs_tensor, acts_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                stats_dict["epoch_num"] = epoch_num
                stats_dict["n_updates"] = batch_num
                stats_dict["batch_size"] = len(trans)

                if batch_num % log_interval == 0:
                    for k, v in stats_dict.items():
                        logger.record(k, v)
                    logger.dump(batch_num)

            if on_epoch_end is not None:
                on_epoch_end(locals())

    def save_policy(self, policy_path: str) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        th.save(self.policy, policy_path)
