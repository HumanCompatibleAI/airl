import datetime
import functools
import itertools
import os
import uuid
from typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import gym
import numpy as np
import stable_baselines3
from gym.wrappers import TimeLimit
from stable_baselines3.common import monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_uuid}"


def make_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
) -> VecEnv:
    """Returns a VecEnv initialized with `n_envs` Envs.

    Args:
        env_name: The Env's string id in Gym.
        n_envs: The number of duplicate environments.
        seed: The environment seed.
        parallel: If True, uses SubprocVecEnv; otherwise, DummyVecEnv.
        log_dir: If specified, saves Monitor output to this directory.
        max_episode_steps: If specified, wraps each env in a TimeLimit wrapper
            with this episode length. If not specified and `max_episode_steps`
            exists for this `env_name` in the Gym registry, uses the registry
            `max_episode_steps` for every TimeLimit wrapper (this automatic
            wrapper is the default behavior when calling `gym.make`). Otherwise
            the environments are passed into the VecEnv unwrapped.
        post_wrappers: If specified, iteratively wraps each environment with each
            of the wrappers specified in the sequence. The argument should be a Callable
            accepting two arguments, the Env to be wrapped and the environment index,
            and returning the wrapped Env.
    """
    # Resolve the spec outside of the subprocess first, so that it is available to
    # subprocesses running `make_env` via automatic pickling.
    spec = gym.spec(env_name)

    def make_env(i, this_seed):
        # Previously, we directly called `gym.make(env_name)`, but running
        # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
        # created a weird interaction between Gym and Ray -- `gym.make` would fail
        # inside this function for any of our custom environment unless those
        # environments were also `gym.register()`ed inside `make_env`. Even
        # registering the custom environment in the scope of `make_vec_env` didn't
        # work. For more discussion and hypotheses on this issue see PR #160:
        # https://github.com/HumanCompatibleAI/imitation/pull/160.
        env = spec.make()

        # Seed each environment with a different, non-sequential seed for diversity
        # (even if caller is passing us sequentially-assigned base seeds). int() is
        # necessary to work around gym bug where it chokes on numpy int64s.
        env.seed(int(this_seed))

        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps)
        elif spec.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=spec.max_episode_steps)

        # Use Monitor to record statistics needed for Baselines algorithms logging
        # Optionally, save to disk
        log_path = None
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "monitor")
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.path.join(log_subdir, f"mon{i:03d}")

        env = monitor.Monitor(env, log_path)

        if post_wrappers:
            for wrapper in post_wrappers:
                env = wrapper(env, i)

        return env

    rng = np.random.RandomState(seed)
    env_seeds = rng.randint(0, (1 << 31) - 1, (n_envs,))
    env_fns = [functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        return DummyVecEnv(env_fns)


def init_rl(
    env: Union[gym.Env, VecEnv],
    model_class: Type[BaseAlgorithm] = stable_baselines3.PPO,
    policy_class: Type[BasePolicy] = ActorCriticPolicy,
    **model_kwargs,
):
    """Instantiates a policy for the provided environment.

    Args:
        env: The (vector) environment.
        model_class: A Stable Baselines RL algorithm.
        policy_class: A Stable Baselines compatible policy network class.
        model_kwargs (dict): kwargs passed through to the algorithm.
          Note: anything specified in `policy_kwargs` is passed through by the
          algorithm to the policy network.

    Returns:
      An RL algorithm.
    """
    # FIXME(sam): verbose=1 and tensorboard_log=None is a hack to prevent SB3
    # from reconfiguring the logger after we've already configured it. Should
    # remove once SB3 issue #109 is fixed (there are also >=2 other comments to
    # this effect elsewhere; worth grepping for "#109").
    all_kwargs = {
        "verbose": 1,
        "tensorboard_log": None,
    }
    all_kwargs.update(model_kwargs)
    return model_class(
        policy_class, env, **all_kwargs
    )  # pytype: disable=not-instantiable


def docstring_parameter(*args, **kwargs):
    """Treats the docstring as a format string, substituting in the arguments."""

    def helper(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return helper


T = TypeVar("T")


def endless_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Generator that endlessly yields elements from iterable.

    If any call to `iter(iterable)` has no elements, then this function raises
    ValueError.

    >>> x = range(2)
    >>> it = endless_iter(x)
    >>> next(it)
    0
    >>> next(it)
    1
    >>> next(it)
    0

    """
    try:
        next(iter(iterable))
    except StopIteration:
        err = ValueError(f"iterable {iterable} had no elements to iterate over.")
        raise err

    return itertools.chain.from_iterable(itertools.repeat(iterable))
