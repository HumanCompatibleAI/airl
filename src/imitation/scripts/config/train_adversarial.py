"""Configuration for imitation.scripts.train_adversarial."""

import os

import sacred

from imitation.policies import base
from imitation.scripts.config.common import DEFAULT_INIT_RL_KWARGS
from imitation.util import util

train_ex = sacred.Experiment("train_adversarial", interactive=True)


@train_ex.config
def train_defaults():
    env_name = "CartPole-v1"  # environment to train on
    total_timesteps = 1e5  # Num of environment transitions to sample
    algorithm = "gail"  # Either "airl" or "gail"

    n_expert_demos = None  # Num demos used. None uses every demo possible
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return

    # Number of environments in VecEnv, must evenly divide gen_batch_size
    num_vec = 8

    # Use SubprocVecEnv rather than DummyVecEnv (generally faster if num_vec>1)
    parallel = True
    max_episode_steps = None  # Set to positive int to limit episode horizons

    # Kwargs for initializing GAIL and AIRL
    algorithm_kwargs = dict(
        shared=dict(
            expert_batch_size=1024,  # Number of expert samples per discriminator update
            # Number of discriminator updates after each round of generator updates
            n_disc_updates_per_round=4,
        ),
        airl={},
        gail={},
    )

    # Kwargs for initializing {GAIL,AIRL}DiscrimNet
    discrim_net_kwargs = dict(shared={}, airl={}, gail={})

    # Modifies the __init__ arguments for the imitation policy
    init_rl_kwargs = dict(
        policy_class=base.FeedForward32Policy,
        **DEFAULT_INIT_RL_KWARGS,
    )
    gen_batch_size = 2048  # Batch size for generator updates

    log_root = os.path.join("output", "train_adversarial")  # output directory
    checkpoint_interval = 0  # Num epochs between checkpoints (<0 disables)
    init_tensorboard = False  # If True, then write Tensorboard logs
    rollout_hint = None  # Used to generate default rollout_path
    data_dir = "data/"  # Default data directory


@train_ex.config
def aliases_default_gen_batch_size(algorithm_kwargs, gen_batch_size):
    # Setting generator buffer capacity and discriminator batch size to
    # the same number is equivalent to not using a replay buffer at all.
    # "Disabling" the replay buffer seems to improve convergence speed, but may
    # come at a cost of stability.

    algorithm_kwargs["shared"]["gen_replay_buffer_capacity"] = gen_batch_size


@train_ex.config
def calc_n_steps(num_vec, gen_batch_size):
    init_rl_kwargs = dict(n_steps=gen_batch_size // num_vec)


@train_ex.config
def paths(env_name, log_root, rollout_hint, data_dir):
    log_dir = os.path.join(
        log_root, env_name.replace("/", "_"), util.make_unique_timestamp()
    )

    # Recommended that user sets rollout_path manually.
    # By default we guess the named config associated with `env_name`
    # and attempt to load rollouts from `data/expert_models/`.
    if rollout_hint is None:
        rollout_hint = env_name.split("-")[0].lower()
    rollout_path = os.path.join(
        data_dir, "expert_models", f"{rollout_hint}_0", "rollouts", "final.pkl"
    )


# Training algorithm named configs


@train_ex.named_config
def gail():
    """Quick alias for algorithm=gail"""
    algorithm = "gail"


@train_ex.named_config
def airl():
    """Quick alias for algorithm=airl"""
    algorithm = "airl"


# Shared settings

MUJOCO_SHARED_LOCALS = dict(discrim_net_kwargs=dict(airl=dict(entropy_weight=0.1)))

ANT_SHARED_LOCALS = dict(
    total_timesteps=3e7,
    max_episode_steps=500,  # To match `inverse_rl` settings.
    algorithm_kwargs=dict(shared=dict(expert_batch_size=8192)),
    gen_batch_size=16384,
)


# Classic RL Gym environment named configs


@train_ex.named_config
def acrobot():
    env_name = "Acrobot-v1"
    rollout_hint = "acrobot"


@train_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    rollout_hint = "cartpole"
    discrim_net_kwargs = {}


@train_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    rollout_hint = "mountain_car"


@train_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"
    rollout_hint = "pendulum"


# Standard MuJoCo Gym environment named configs


@train_ex.named_config
def ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    env_name = "Ant-v2"
    rollout_hint = "ant"


@train_ex.named_config
def half_cheetah():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "HalfCheetah-v2"
    rollout_hint = "half_cheetah"
    total_timesteps = 2e6


@train_ex.named_config
def hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    # TODO(adam): upgrade to Hopper-v3?
    env_name = "Hopper-v2"
    rollout_hint = "hopper"


@train_ex.named_config
def humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "Humanoid-v2"
    rollout_hint = "humanoid"
    total_timesteps = 4e6


@train_ex.named_config
def reacher():
    env_name = "Reacher-v2"
    rollout_hint = "reacher"


@train_ex.named_config
def swimmer():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "Swimmer-v2"
    rollout_hint = "swimmer"
    total_timesteps = 2e6


@train_ex.named_config
def walker():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "Walker2d-v2"
    rollout_hint = "walker"


# Custom Gym environment named configs


@train_ex.named_config
def two_d_maze():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "imitation/TwoDMaze-v0"
    rollout_hint = "two_d_maze"


@train_ex.named_config
def custom_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    # Watch out -- ANT_SHARED_LOCALS could erroneously erase nested dict keys from
    # MUJOCO_SHARED_LOCALS because `locals().update()` doesn't merge dicts
    # "Sacred-style".
    locals().update(**ANT_SHARED_LOCALS)
    env_name = "imitation/CustomAnt-v0"
    rollout_hint = "custom_ant"


@train_ex.named_config
def disabled_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    env_name = "imitation/DisabledAnt-v0"
    rollout_hint = "disabled_ant"


# Debug configs


@train_ex.named_config
def fast():
    """Minimize the amount of computation.

    Useful for test cases.
    """
    # Need a minimum of 10 total_timesteps for adversarial training code to pass
    # "any update happened" assertion inside training loop.
    total_timesteps = 10
    n_expert_demos = 1
    n_episodes_eval = 1
    algorithm_kwargs = dict(
        shared=dict(
            expert_batch_size=1,
            n_disc_updates_per_round=4,
        )
    )
    gen_batch_size = 2
    parallel = False  # easier to debug with everything in one process
    max_episode_steps = 5
    # SB3 RL seems to need batch size of 2, otherwise it runs into numeric
    # issues when computing multinomial distribution during predict()
    num_vec = 2
    init_rl_kwargs = dict(batch_size=2)
