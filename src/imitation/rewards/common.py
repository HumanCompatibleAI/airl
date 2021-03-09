"""Utilities and definitions shared by reward-related code."""

import collections
import functools
from typing import Callable, Dict, List, Tuple

import gym
import numpy as np
import torch as th
from stable_baselines3.common import preprocessing, vec_env

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _reward_fn_normalize_inputs(
    obs: np.ndarray,
    acts: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    *,
    reward_fn: RewardFn,
    vec_normalize: vec_env.VecNormalize,
    norm_reward: bool = True,
) -> np.ndarray:
    """Combine with `functools.partial` to create an input-normalizing RewardFn.

    Args:
        reward_fn: The reward function that normalized inputs are evaluated on.
        vec_normalize: Instance of VecNormalize used to normalize inputs and
            rewards.
        norm_reward: If True, then also normalize reward before returning.

    Returns:
        The possibly normalized reward.
    """
    norm_obs = vec_normalize.normalize_obs(obs)
    norm_next_obs = vec_normalize.normalize_obs(next_obs)
    rew = reward_fn(norm_obs, acts, norm_next_obs, dones)
    if norm_reward:
        rew = vec_normalize.normalize_reward(rew)
    return rew


def build_norm_reward_fn(*, reward_fn, vec_normalize, **kwargs) -> RewardFn:
    """Reward function that automatically normalizes inputs.

    See _reward_fn_normalize_inputs for argument documentation.
    """
    return functools.partial(
        _reward_fn_normalize_inputs,
        reward_fn=reward_fn,
        vec_normalize=vec_normalize,
        **kwargs,
    )


def disc_rew_preprocess_inputs(
    observation_space: gym.Space,
    action_space: gym.Space,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    done: np.ndarray,
    device: th.device,
    *,
    normalize_images: bool,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    state_th = th.as_tensor(state, device=device)
    action_th = th.as_tensor(action, device=device)
    next_state_th = th.as_tensor(next_state, device=device)
    done_th = th.as_tensor(done, device=device)

    del state, action, next_state, done  # unused

    # preprocess
    state_th = preprocessing.preprocess_obs(
        state_th, observation_space, normalize_images=normalize_images
    )
    action_th = preprocessing.preprocess_obs(
        action_th, action_space, normalize_images=normalize_images
    )
    next_state_th = preprocessing.preprocess_obs(
        next_state_th, observation_space, normalize_images=normalize_images
    )
    done_th = done_th.to(th.float32)

    n_gen = len(state_th)
    assert state_th.shape == next_state_th.shape
    assert len(action_th) == n_gen

    return state_th, action_th, next_state_th, done_th


def compute_train_stats(
    disc_logits_gen_is_high: th.Tensor,
    labels_gen_is_one: th.Tensor,
    disc_loss: th.Tensor,
) -> Dict[str, float]:
    """Train statistics for GAIL/AIRL discriminator, or other binary classifiers.

    Args:
        disc_logits_gen_is_high: discriminator logits produced by
            `DiscrimNet.logits_gen_is_high`.
        labels_gen_is_one: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.

    Returns:
        stats: dictionary mapping statistic names for float values."""
    with th.no_grad():
        bin_is_generated_pred = disc_logits_gen_is_high > 0
        bin_is_generated_true = labels_gen_is_one > 0
        bin_is_expert_true = th.logical_not(bin_is_generated_true)
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()
        n_generated = float(th.sum(int_is_generated_true))
        n_labels = float(len(labels_gen_is_one))
        n_expert = n_labels - n_generated
        pct_expert = n_expert / float(n_labels) if n_labels > 0 else float("NaN")
        n_expert_pred = int(n_labels - th.sum(int_is_generated_pred))
        if n_labels > 0:
            pct_expert_pred = n_expert_pred / float(n_labels)
        else:
            pct_expert_pred = float("NaN")
        correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = th.mean(correct_vec.float())

        _n_pred_expert = th.sum(th.logical_and(bin_is_expert_true, correct_vec))
        if n_expert < 1:
            expert_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Python ints
            expert_acc = _n_pred_expert / float(n_expert)

        _n_pred_gen = th.sum(th.logical_and(bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / float(_n_gen_or_1)

        label_dist = th.distributions.Bernoulli(th.sigmoid(disc_logits_gen_is_high))
        entropy = th.mean(label_dist.entropy())

    pairs = [
        ("disc_loss", float(th.mean(disc_loss))),
        # accuracy, as well as accuracy on *just* expert examples and *just*
        # generated examples
        ("disc_acc", float(acc)),
        ("disc_acc_expert", float(expert_acc)),
        ("disc_acc_gen", float(generated_acc)),
        # entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        ("disc_entropy", float(entropy)),
        # true number of expert demos and predicted number of expert demos
        ("disc_proportion_expert_true", float(pct_expert)),
        ("disc_proportion_expert_pred", float(pct_expert_pred)),
        ("n_expert", float(n_expert)),
        ("n_generated", float(n_generated)),
    ]  # type: List[Tuple[str, float]]
    return collections.OrderedDict(pairs)
