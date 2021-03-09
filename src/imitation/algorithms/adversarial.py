import dataclasses
import logging
import os
from typing import Callable, Dict, Iterable, Mapping, Optional, Type, Union

import gym
import numpy as np
import torch as th
import torch.utils.data as th_data
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import on_policy_algorithm, preprocessing, vec_env, callbacks as sb3_callbacks

from imitation.data import buffer, types, wrappers
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, reward_wrapper, util


class AdversarialTrainer:
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    venv: vec_env.VecEnv
    """The original vectorized environment."""

    venv_norm_obs: vec_env.VecEnv
    """Like `self.venv`, but wrapped with `VecNormalize` normalizing the observations.

    These statistics must be saved along with the model."""

    venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    def __init__(
        self,
        venv: vec_env.VecEnv,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        discrim: discrim_nets.DiscrimNet,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        n_disc_updates_per_round: int = 2,
        *,
        log_dir: str = "output/",
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        normalize_reward_std: float = 1.0,
        clip_obs: float = float("inf"),
        clip_reward: float = float("inf"),
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
        gen_replay_buffer_capacity: Optional[int] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        disc_augmentation_fn: Callable[[th.Tensor], th.Tensor] = None,
        gen_callbacks: Optional[Iterable[sb3_callbacks.BaseCallback]] = None,
    ):
        """Builds AdversarialTrainer.

        Args:
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. The generator batch size
                `self.gen_batch_size` is inferred from `gen_algo.n_steps`.
            discrim: The discriminator network. This will be moved to the same
                device as `gen_algo`.
            expert_data: Either a `torch.utils.data.DataLoader`-like object or an
                instance of `Transitions` which is automatically converted into a
                shuffled version of the former type.

                If the argument passed is a `DataLoader`, then it must yield batches of
                expert data via its `__iter__` method. Each batch is a dictionary whose
                keys "obs", "acts", "next_obs", and "dones", correspond to Tensor or
                NumPy array values each with batch dimension equal to
                `expert_batch_size`. If any batch dimension doesn't equal
                `expert_batch_size` then a `ValueError` is raised.

                If the argument is a `Transitions` instance, then `len(expert_data)`
                must be at least `expert_batch_size`.
            expert_batch_size: The number of samples in each batch yielded from
                the expert data loader. The discriminator batch size is twice this
                number because each discriminator batch contains a generator sample for
                every expert sample.
            n_discrim_updates_per_round: The number of discriminator updates after each
                round of generator updates in AdversarialTrainer.learn().
            log_dir: Directory to store TensorBoard logs, plots, etc. in.
            normalize_obs: Whether to normalize observations with `VecNormalize`.
            normalize_reward: Whether to normalize rewards with `VecNormalize`.
            normalize_reward_std: Target standard deviation of rewards.
            clip_obs: Bound on l_infty magnitude of observation.
            clip_reward: Bound on magnitude of reward.
            disc_opt_cls: The optimizer for discriminator training.
            disc_opt_kwargs: Parameters for discriminator training.
            gen_replay_buffer_capacity: The capacity of the
                generator replay buffer (the number of obs-action-obs samples from
                the generator that can be stored).

                By default this is equal to `self.gen_batch_size`, meaning that we
                sample only from the most recent batch of generator samples.
            init_tensorboard: If True, makes various discriminator
                TensorBoard summaries.
            init_tensorboard_graph: If both this and `init_tensorboard` are True,
                then write a Tensorboard graph summary to disk.
            debug_use_ground_truth: If True, use the ground truth reward for
                `self.train_env`.
                This disables the reward wrapping that would normally replace
                the environment reward with the learned reward. This is useful for
                sanity checking that the policy training is functional.
            disc_augmentation_fn: Function to augment a batch of (on-device)
                discriminator training inputs (default: identity).
            gen_callbacks: SB3 callbacks to give to `PPO.learn()` during
                generator training.
        """

        assert (
            logger.is_configured()
        ), "Requires call to imitation.util.logger.configure"
        self._global_step = 0
        self._disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round

        if expert_batch_size <= 0:
            raise ValueError(f"expert_batch_size={expert_batch_size} must be positive.")

        self.expert_batch_size = expert_batch_size
        if isinstance(expert_data, types.Transitions):
            if len(expert_data) < expert_batch_size:
                raise ValueError(
                    "Provided Transitions instance as `expert_data` argument but "
                    "len(expert_data) < expert_batch_size. "
                    f"({len(expert_data)} < {expert_batch_size})."
                )

            self.expert_data_loader = th_data.DataLoader(
                expert_data,
                batch_size=expert_batch_size,
                collate_fn=types.transitions_collate_fn,
                shuffle=True,
                drop_last=True,
            )
        else:
            self.expert_data_loader = expert_data
        self._endless_expert_iterator = util.endless_iter(self.expert_data_loader)

        if disc_augmentation_fn is None:
            disc_augmentation_fn = util.identity
        self.disc_augmentation_fn = disc_augmentation_fn
        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo
        self._log_dir = log_dir

        # Create graph for optimising/recording stats on discriminator
        self.discrim = discrim.to(self.gen_algo.device)
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._disc_opt = self._disc_opt_cls(
            self.discrim.parameters(), **self._disc_opt_kwargs
        )

        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        self.venv_buffering = wrappers.BufferingWrapper(self.venv)
        self.venv_norm_obs = vec_env.VecNormalize(
            self.venv_buffering,
            norm_reward=False,
            norm_obs=normalize_obs,
            clip_obs=clip_obs,
            clip_reward=None,
            norm_reward_std=None,
        )

        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.venv_wrapped = self.venv_norm_obs
            all_gen_callbacks = []
        else:
            self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
                self.venv_norm_obs, self.discrim.predict_reward_train
            )
            all_gen_callbacks = [self.venv_wrapped.make_log_callback()]
        if gen_callbacks:
            all_gen_callbacks.extend(gen_callbacks)
        self.gen_callback = sb3_callbacks.CallbackList(all_gen_callbacks)

        self.venv_train = vec_env.VecNormalize(
            self.venv_wrapped,
            norm_obs=False,
            norm_reward=normalize_reward,
            norm_reward_std=normalize_reward_std,
            clip_reward=clip_reward,
            clip_obs=float("inf"),
        )

        self.gen_algo.set_env(self.venv_train)

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_batch_size
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity, self.venv
        )

    def _next_expert_batch(self) -> Mapping:
        return next(self._endless_expert_iterator)

    @property
    def gen_batch_size(self) -> int:
        return self.gen_algo.n_steps * self.gen_algo.get_env().num_envs

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
        dump_logs: bool = False,
    ) -> Dict[str, float]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.expert_batch_size` samples.

                If this argument is not provided, then `self.expert_batch_size` expert
                samples from `self.expert_data_loader` are used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.expert_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.
            dump_logs: Whether to dump discriminator training logs on every iteration.

        Returns:
           dict: Statistics for discriminator (e.g. loss, accuracy).
        """
        with logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            batch = self._make_disc_train_batch(
                gen_samples=gen_samples, expert_samples=expert_samples
            )
            disc_logits = self.discrim.logits_gen_is_high(
                batch["state"],
                batch["action"],
                batch["next_state"],
                batch["done"],
                batch["log_policy_act_prob"],
            )
            loss = self.discrim.disc_loss(disc_logits, batch["labels_gen_is_one"])

            # do gradient step
            self._disc_opt.zero_grad()
            loss.backward()
            self._disc_opt.step()
            self._disc_step += 1

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = rew_common.compute_train_stats(
                    disc_logits, batch["labels_gen_is_one"], loss
                )
            logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                logger.record_mean(k, v)
            if dump_logs:
                logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ):
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
          total_timesteps: The number of transitions to sample from
            `self.venv_train` during training. By default,
            `self.gen_batch_size`.
          learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
            method.
        """
        if total_timesteps is None:
            total_timesteps = self.gen_batch_size
        if learn_kwargs is None:
            learn_kwargs = {}

        with logger.accumulate_means("gen"):
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                **learn_kwargs,
            )
            self._global_step += 1

        gen_samples = self.venv_buffering.pop_transitions()
        self._gen_replay_buffer.store(gen_samples)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
        log_interval_timesteps: Optional[int] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.

        Every "round" consists of a call to `train_gen(self.gen_batch_size)`,
        a call to `train_disc`, and finally a call to `callback(round)`.

        Training ends once an additional "round" would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.

        Args:
          total_timesteps: An upper bound on the number of transitions to sample
              from the environment during training.
          callback: A function called at the end of every round which takes in a
              single argument, the round number. Round numbers are in
              `range(total_timesteps // self.gen_batch_size)`.
          log_interval_timesteps: dump logs every `log_interval_timesteps`. If
              None, will dump as frequently as possible.
        """
        n_rounds = total_timesteps // self.gen_batch_size
        last_log_dump = None
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_batch_size} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for r in tqdm.trange(n_rounds, desc="round"):
            self.train_gen(
                self.gen_batch_size,
                learn_kwargs=dict(dump_logs=log_interval_timesteps is None,
                                  progress_max_timesteps=total_timesteps))
            for _ in range(self.n_disc_updates_per_round):
                self.train_disc(dump_logs=log_interval_timesteps is None)
            if callback:
                callback(r)
            num_timesteps = self.gen_algo.num_timesteps
            if (last_log_dump is None or log_interval_timesteps is None or
               num_timesteps > last_log_dump + log_interval_timesteps):
                last_log_dump = num_timesteps
                logger.dump(self._global_step)

    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device=self.discrim.device(), **kwargs)

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        # always make it contiguous so that we can feed it to augmentation
        # module, RNNs, etc.
        tensor = th.as_tensor(
            ndarray, device=self.discrim.device(), **kwargs
        ).contiguous()
        preprocessed = preprocessing.preprocess_obs(
            tensor,
            space,
            normalize_images=self.discrim.normalize_images,
        )
        return preprocessed

    def _make_disc_train_batch(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> dict:
        """Build and return training batch for the next discriminator update.

        Args:
          gen_samples: Same as in `train_disc_step`.
          expert_samples: Same as in `train_disc_step`.
        """
        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first."
                )
            gen_samples = self._gen_replay_buffer.sample(self.expert_batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples)

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])
        if not (n_gen == n_expert == self.expert_batch_size):
            raise ValueError(
                "Need to have exactly self.expert_batch_size number of expert and "
                "generator samples, each. "
                f"(n_gen={n_gen} n_expert={n_expert} "
                f"expert_batch_size={self.expert_batch_size})"
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        n_samples = n_expert + n_gen
        assert n_expert == len(expert_samples["acts"])
        assert n_expert == len(expert_samples["next_obs"])
        assert n_gen == len(gen_samples["acts"])
        assert n_gen == len(gen_samples["next_obs"])

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        labels_gen_is_one = np.concatenate(
            [np.zeros(n_expert, dtype=int), np.ones(n_gen, dtype=int)]
        )
        # Policy and reward network were trained on normalized observations.
        obs = self.venv_norm_obs.normalize_obs(obs)
        next_obs = self.venv_norm_obs.normalize_obs(next_obs)

        # Calculate generator-policy log probabilities.
        # FIXME(sam): skip this if we don't actually need log probabilities
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.gen_algo.device)
            acts_th = th.as_tensor(acts, device=self.gen_algo.device)
            _, log_act_prob_th, _ = self.gen_algo.policy.evaluate_actions(
                obs_th, acts_th
            )
            log_act_prob = log_act_prob_th.detach().cpu().numpy()
            del obs_th, acts_th, log_act_prob_th  # unneeded
        assert len(log_act_prob) == n_samples
        log_act_prob = log_act_prob.reshape((n_samples,))

        # torchify states first so that we can apply augmentations
        with th.no_grad():
            torch_state = self._torchify_with_space(obs, self.discrim.observation_space)
            torch_next_state = self._torchify_with_space(
                next_obs, self.discrim.observation_space
            )
        torch_state = self.disc_augmentation_fn(torch_state)
        torch_next_state = self.disc_augmentation_fn(torch_next_state)

        batch_dict = {
            "state": torch_state,
            "next_state": torch_next_state,
            "action": self._torchify_with_space(acts, self.discrim.action_space),
            "done": self._torchify_array(dones),
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
            "log_policy_act_prob": self._torchify_array(log_act_prob),
        }

        return batch_dict


class GAIL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        *,
        # FIXME(sam) pass in discrim net directly; don't ask for kwargs indirectly
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning.

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `GAIL` adds on top of its superclass initializer are
        as follows:

        Args:
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetGAIL.

        """
        discrim_kwargs = discrim_kwargs or {}
        discrim = discrim_nets.DiscrimNetGAIL(
            venv.observation_space, venv.action_space, **discrim_kwargs
        )
        super().__init__(
            venv, gen_algo, discrim, expert_data, expert_batch_size, **kwargs
        )


class AIRL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        *,
        # FIXME(sam): pass in reward net directly, not via _cls and _kwargs
        reward_net_cls: Type[reward_nets.RewardNet] = reward_nets.BasicShapedRewardNet,
        reward_net_kwargs: Optional[Mapping] = None,
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Adversarial Inverse Reinforcement Learning.

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `AIRL` adds on top of its superclass initializer are
        as follows:

        Args:
            reward_net_cls: Reward network constructor. The reward network is part of
                the AIRL discriminator.
            reward_net_kwargs: Optional keyword arguments to use while constructing
                the reward network.
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetAIRL.
        """
        # TODO(shwang): Maybe offer str=>RewardNet conversion like
        #  stable_baselines3 does with policy classes.
        reward_net_kwargs = reward_net_kwargs or {}
        reward_network = reward_net_cls(
            action_space=venv.action_space,
            observation_space=venv.observation_space,
            # pytype is afraid that we'll directly call RewardNet() which is an abstract
            # class, hence the disable.
            **reward_net_kwargs,  # pytype: disable=not-instantiable
        )

        discrim_kwargs = discrim_kwargs or {}
        discrim = discrim_nets.DiscrimNetAIRL(reward_network, **discrim_kwargs)
        super().__init__(
            venv, gen_algo, discrim, expert_data, expert_batch_size, **kwargs
        )
