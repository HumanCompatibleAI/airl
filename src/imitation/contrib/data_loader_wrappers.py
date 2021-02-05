import abc
import warnings
from typing import Any, Generic, Iterable, Iterator, List, Mapping, TypeVar

import gym
import numpy as np
import torch as th

S = TypeVar("S")
T = TypeVar("T")


class DataLoaderWrapper(Generic[S, T], Iterable[T], abc.ABC):
    def __init__(self, data_loader: Iterable[S]):
        self.data_loader = data_loader

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T]:
        pass


class ApplyDataLoaderWrapperMixin(abc.ABC, Generic[S, T]):
    """Concrete instances of this abstract mixin define a `DatasetWrapper`.

    Intended for use as an abstract mix-in in a subclass of `gym.Wrapper` so that the
    `gym.Wrapper` can be processed by `wrap_data_loader_with_env_wrappers`.
    """

    @abc.abstractmethod
    def apply_data_loader_wrapper(
        self,
        data_loader: Iterable[Mapping],
    ) -> DataLoaderWrapper[S, T]:
        """Wraps a dataset using the dataset.

        Args:
            data_loader: A DataLoader to wrap.

        Returns:
            A wrapped data_loader.
        """


class Transform(abc.ABC):

    """
    Abstract class for simple batched transformations on environment transitions,
    suitable for building a simple `gym.Wrapper` with an analogous `DataLoaderWrapper`.
    (In other words, `GymWrapperFromTransform` and `DataLoaderWrapperFromTransform`.)

    Abstract methods in this class are prefixed up_* and down_* using a stack-like
    convention for wrappers. The outermost wrapper is at the top of the "stack" and when
    moving to inner wrappers, we go "down the stack" until we reach the `gym.Env` or the
    `DataLoader`. Likewise we can start from the `gym.Env` and `DataLoader` and go "up
    the stack" until the transitions are forwarded to the policy or algorithm.

    All of the abstract methods are prefixed up_* except for `down_acts`, which is
    only used in derivative Gym wrappers to pass modified actions to the environment.
    All other transformations can only occur when transitions are moving up the stack.
    """

    @abc.abstractmethod
    def down_acts(self, acts: Any) -> Any:
        """Transformation applied to actions as they descend through a Gym.Wrapper.

        A Gym.Wrapper derived from this transform takes an action from the policy or a
        higher wrapper, modifies the action with `down_acts`, and then passes the
        transformed action one layer deeper to the lower wrapper or Env.

        DataLoaderWrappers derived from this transform don't use `down_acts`. They use
        `up_acts` instead.

        Args:
            acts: Batched actions.

        Returns:
            A transformed action which is either consumed by wrapped `Env`, or passed
            deeper to the next Gym Wrapper.
        """

    @abc.abstractmethod
    def up_acts(self, acts: Any) -> Any:
        """Transformation applied to actions as they ascend through a DataLoaderWrapper.

        Gym.Wrappers derived from this Transform don't use `up_acts`.

        A DataLoaderWrapper derived from this Transform takes an action from a
        Torch-style DataLoader or a lower wrapper, modifies the action with `up_acts`,
        and then passes the transformed action one layer higher to the higher wrapper
        or the user.

        Args:
            acts: Batched actions.

        Returns:
            A transformed action which is either consumed by the policy, or passed
            higher to the next DataLoaderWrapper.
        """

    @abc.abstractmethod
    def up_obs(self, obs: Any) -> Any:
        """Transformation applied to observations when ascending both types of wrappers.

        Both Gym.Wrapper and DataLoaderWrappers derived from this Transform
        apply this transformation when passing observations from a lower layer (either
        another wrapper, or the source Env/DataLoader) to a higher layer (either another
        wrapper, or the policy).
        """

    @abc.abstractmethod
    def up_rews(self, rews: np.ndarray) -> np.ndarray:
        """Transformation applied to rewards when ascending both types of wrappers.

        Analogous to `up_obs` but for rewards.
        """


class DataLoaderWrapperFromTransform(DataLoaderWrapper[dict, dict]):
    """DataLoaderWrapper that corresponds to an instance of Transform.

    Primarily used by GymWrapperFromTransform.
    """

    def __init__(
        self,
        data_loader: Iterable[Mapping],
        transform: Transform,
    ):
        self.data_loader = data_loader
        self.transform = transform
        super().__init__(data_loader)

    def __iter__(self) -> Iterator[dict]:
        for trans_dict in self.data_loader:
            result = dict(trans_dict)
            result["obs"] = self.transform.up_obs(result["obs"])
            result["acts"] = self.transform.up_acts(result["acts"])
            if "next_obs" in result:
                result["next_obs"] = self.transform.up_obs(result["next_obs"])
            if "rews" in result:
                result["rews"] = self.transform.up_obs(result["rews"])
            yield result


# XXX: Maybe provide VecEnvGymWrapperFromTransform in addition or as a replacement?
# For our current use cases we only have been fiddling with non-VecEnv Wrappers.


class GymWrapperFromTransform(
    gym.Wrapper,
    ApplyDataLoaderWrapperMixin[dict, dict],
):
    """Simple Gym Wrapper defined by an instance of BidirectionalTransform.

    It trivially implements `DefinesDataLoaderWrapper.apply_data_loader_wrapper`
    using the same `BidirectionalTransform`.
    """

    def __init__(self, env: gym.Env, transform: Transform):
        self.transform = transform
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        return self.forward_obs(obs)

    def step(self, action: Any):
        act = self.transform.down_acts(np.array([action]))[0]
        ob, rew, done, info = self.env.step(act)
        upped_ob = self.transform.up_obs(np.array([ob]))[0]
        upped_rew = self.transform.up_rews(np.array([rew]))[0]
        return upped_ob, upped_rew, done, info

    def apply_data_loader_wrapper(
        self,
        data_loader: Iterable[Mapping],
    ) -> DataLoaderWrapper[dict, dict]:
        return DataLoaderWrapperFromTransform(data_loader, self.transform)


def wrap_dataset_with_env_wrappers(
    data_loader: Iterable[Mapping],
    env: gym.Env,
    warn_on_ignore_wrapper: bool = True,
) -> Iterable:
    """Apply DataLoaderWrappers corresponding to each gym.Wrapper on `env`.

    Args:
        data_loader: Base `DataLoaderInterface` instance to wrap with
            DataLoaderWrappers corresponding to each gym.Wrapper.
        env: For every gym.Wrapper `wrapper` wrapping `env` which also subclasses
            `ApplyDataLoaderWrapperMixin`, `data_loader` is wrapped using
            `wrapper.apply_data_loader_wrapper()`.

            Any `gym.Wrapper` that doesn't subclass `ApplyDataLoaderWrapperMixin` is
            skipped.
        warn_on_ignore_wrapper: If True, then warn with RuntimeWarning for every
            `gym.Wrapper` that doesn't subclass `ApplyDataLoaderWrapperMixin`.

    Returns:
        `data_loader` wrapped by a new `DatasetWrapper` for every `gym.Wrapper` around
        `env` also subclasses `DefinesDataLoaderWrapper`. If `env` has no wrappers,
        then the return is simply `data_loader`.
    """

    # Traverse all the gym.Wrapper starting from the outermost wrapper. When re-apply
    # them to the Dataset as MineRLDatasetWrapper, we need to apply in reverse order.

    compatible_wrappers: List[ApplyDataLoaderWrapperMixin] = []
    curr = env
    while isinstance(curr, gym.Wrapper):
        if isinstance(curr, ApplyDataLoaderWrapperMixin):
            compatible_wrappers.append(curr)
        elif warn_on_ignore_wrapper:
            warnings.warn(
                f"{curr} doesn't subclass DefinesDatasetWrapperMixin. Skipping "
                "this Wrapper when creating Dataset processing wrappers.",
                RuntimeWarning,
            )
        curr = curr.env

    for wrapper in reversed(compatible_wrappers):
        data_loader = wrapper.apply_data_loader_wrapper(data_loader)
    return data_loader


class TensorToNumpyDataLoaderWrapper(DataLoaderWrapper[dict, dict]):
    """For dict batch, make new dict where every formerly Tensor value is a Numpy array.

    All other values are left the same.
    """

    def __init__(self, data_loader: Iterable[Mapping]):
        super().__init__(data_loader)

    def __iter__(self) -> Iterator[dict]:
        # TODO(shwang): Handle dict-valued obs, act? (MineRL)
        for x in self.data_loader:
            result = dict(x)
            for k, v in result.items():
                if isinstance(v, th.Tensor):
                    result[k] = v.detach().numpy()
            yield result
