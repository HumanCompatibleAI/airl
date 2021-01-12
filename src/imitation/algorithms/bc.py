"""Behavioural Cloning (BC).

Trains policy by applying supervised learning to a fixed dataset of (observation,
action) pairs generated by some expert demonstrator.
"""

import contextlib
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import torch.utils.data as th_data
import tqdm.autonotebook as tqdm
from stable_baselines3.common import logger, policies, utils

from imitation.data import types
from imitation.policies import base


def reconstruct_policy(
    policy_path: str,
    device: Union[th.device, str] = "auto",
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


class EpochOrBatchIteratorWithProgress:
    def __init__(
        self,
        data_loader: Iterable[dict],
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
    ):
        """Wraps DataLoader so that all BC batches can be processed in a one for-loop.

        Also uses `tqdm` to show progress in stdout.

        Args:
            data_loader: An iterable over data dicts, as used in `BC`.
            n_epochs: The number of epochs to iterate through in one call to
                __iter__. Exactly one of `n_epochs` and `n_batches` should be provided.
            n_batches: The number of batches to iterate through in one call to
                __iter__. Exactly one of `n_epochs` and `n_batches` should be provided.
            on_epoch_end: A callback function without parameters to be called at the
                end of every epoch.
        """
        if n_epochs is not None and n_batches is None:
            self.use_epochs = True
        elif n_epochs is None and n_batches is not None:
            self.use_epochs = False
        else:
            raise ValueError(
                "Must provide exactly one of `n_epochs` and `n_batches` arguments."
            )

        self.data_loader = data_loader
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.on_epoch_end = on_epoch_end

    def __iter__(self) -> Iterable[Tuple[dict, dict]]:
        """Yields batches while updating tqdm display to display progress."""

        samples_so_far = 0
        epoch_num = 0
        batch_num = 0
        batch_suffix = epoch_suffix = ""
        if self.use_epochs:
            display = tqdm.tqdm(total=self.n_epochs)
            epoch_suffix = f"/{self.n_epochs}"
        else:  # Use batches.
            display = tqdm.tqdm(total=self.n_batches)
            batch_suffix = f"/{self.n_batches}"

        def update_desc():
            display.set_description(
                f"batch: {batch_num}{batch_suffix}  epoch: {epoch_num}{epoch_suffix}"
            )

        with contextlib.closing(display):
            while True:
                update_desc()
                for batch in self.data_loader:
                    batch_num += 1
                    batch_size = len(batch["obs"])
                    assert batch_size > 0
                    samples_so_far += batch_size
                    stats = dict(
                        epoch_num=epoch_num,
                        batch_num=batch_num,
                        samples_so_far=samples_so_far,
                    )
                    yield batch, stats
                    if not self.use_epochs:
                        update_desc()
                        display.update(1)
                        if batch_num >= self.n_batches:
                            return
                epoch_num += 1
                if self.on_epoch_end is not None:
                    self.on_epoch_end()

                if self.use_epochs:
                    update_desc()
                    display.update(1)
                    if epoch_num >= self.n_epochs:
                        return


class BC:

    DEFAULT_BATCH_SIZE: int = 32
    """Default batch size for DataLoader automatically constructed from Transitions.

    See `set_expert_data_loader()`.
    """

    # TODO(scottemmons): pass BasePolicy into BC directly (rather than passing its
    #  arguments)
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Type[policies.BasePolicy],
        expert_data: Union[Iterable[Mapping], types.TransitionsMinimal, None] = None,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, th.device] = "auto",
    ):
        """Behavioral cloning (BC).

        Recovers a policy via supervised learning on observation-action Tensor
        pairs, sampled from a Torch DataLoader or any Iterator that ducktypes
        `torch.utils.data.DataLoader`.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            policy: the policy to be trained.
            expert_data: If not None, then immediately call
                  `self.set_expert_data_loader(expert_data)` during initialization.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                  weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.
        """
        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")

        self.action_space = action_space
        self.observation_space = observation_space
        self.device = device = utils.get_device(device)
        self.device = utils.get_device(device)

        self.policy = policy.policy.to(self.device)
# pytype: disable=not-instantiable
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(self.policy.policy.parameters(), **optimizer_kwargs)

        self.expert_data_loader: Optional[Iterable[Mapping]] = None
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

        if expert_data is not None:
            self.set_expert_data_loader(expert_data)

    def set_expert_data_loader(
        self,
        expert_data: Union[Iterable[Mapping], types.TransitionsMinimal],
    ) -> None:
        """Set the expert data loader, which yields batches of obs-act pairs.

        Changing the expert data loader on-demand is useful for DAgger and other
        interactive algorithms.

        Args:
             expert_data: Either a Torch `DataLoader`, any other iterator that
                yields dictionaries containing "obs" and "acts" Tensors or Numpy arrays,
                or a `TransitionsMinimal` instance.

                If this is a `TransitionsMinimal` instance, then it is automatically
                converted into a shuffled `DataLoader` with batch size
                `BC.DEFAULT_BATCH_SIZE`.
        """
        if isinstance(expert_data, types.TransitionsMinimal):
            self.expert_data_loader = th_data.DataLoader(
                expert_data,
                shuffle=True,
                batch_size=BC.DEFAULT_BATCH_SIZE,
                collate_fn=types.transitions_collate_fn,
            )
        else:
            self.expert_data_loader = expert_data

    def _calculate_loss(
        self,
        obs: Union[th.Tensor, np.ndarray],
        acts: Union[th.Tensor, np.ndarray],
    ) -> Tuple[th.Tensor, Dict[str, float]]:
        """
        Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            obs: The observations seen by the expert. If this is a Tensor, then
                gradients are detached first before loss is calculated.
            acts: The actions taken by the expert. If this is a Tensor, then its
                gradients are detached first before loss is calculated.

        Returns:
            loss: The supervised learning loss for the behavioral clone to optimize.
            stats_dict: Statistics about the learning process to be logged.

        """
        obs = th.as_tensor(obs, device=self.device).detach()
        acts = th.as_tensor(acts, device=self.device).detach()

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
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Callable[[], None] = None,
        log_interval: int = 100,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            on_epoch_end: Optional callback with no parameters to run at the end of each
                epoch.
            log_interval: Log stats after every log_interval batches.
        """
        it = EpochOrBatchIteratorWithProgress(
            self.expert_data_loader,
            n_epochs=n_epochs,
            n_batches=n_batches,
            on_epoch_end=on_epoch_end,
        )

        batch_num = 0
        for batch, stats_dict_it in it:
            loss, stats_dict_loss = self._calculate_loss(batch["obs"], batch["acts"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_num % log_interval == 0:
                for stats in [stats_dict_it, stats_dict_loss]:
                    for k, v in stats.items():
                        logger.record(k, v)
                logger.dump(batch_num)
            batch_num += 1

    def save_policy(self, policy_path: str) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        th.save(self.policy, policy_path)
