"""Configuration for imitation.scripts.train_adversarial."""

import os

import sacred
from stable_baselines.common import policies

from imitation.policies import base
from imitation.scripts.config.common import DEFAULT_INIT_RL_KWARGS
from imitation.util import util

train_ex = sacred.Experiment("train_adversarial", interactive=True)


@train_ex.config
def train_defaults():
    env_name = "CartPole-v1"  # environment to train on
    total_timesteps = 1e5  # Num of environment transitions to sample

    n_expert_demos = None  # Num demos used. None uses every demo possible
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return
    airl_entropy_weight = 1.0

    # Number of epochs in between plots (<0 disables) (=0 means final plot only)
    plot_interval = -1
    n_plot_episodes = 5  # Number of rollouts for each mean_ep_rew data
    # Interval for extra episode rew data. (<=0 disables)
    extra_episode_data_interval = -1
    show_plots = True  # Show plots in addition to saving them

    init_trainer_kwargs = dict(
        num_vec=8,  # Must evenly divide gen_batch_size
        parallel=True,  # Use SubprocVecEnv (generally faster if num_vec>1)
        max_episode_steps=None,  # Set to positive int to limit episode horizons
        scale=True,
        reward_kwargs=dict(theta_units=[32, 32], phi_units=[32, 32],),
        init_rl_kwargs=dict(
            policy_class=base.FeedForward32Policy, **DEFAULT_INIT_RL_KWARGS
        ),
    )

    log_root = os.path.join("output", "train_adversarial")  # output directory
    checkpoint_interval = 0  # num epochs between checkpoints (<0 disables)
    init_tensorboard = False  # If True, then write Tensorboard logs.
    rollout_hint = None  # Used to generate default rollout_path
    data_dir = "data/"  # Default data directory

    # `gen_batch_size` must be a multiple of `init_trainer_kwargs.num_vec`.
    # (If using PPO2, then also must be a multiple of
    # `init_trainer_kwargs.init_rl_kwargs.nminibatch`).
    disc_batch_size = 2048  # Batch size for discriminator updates.
    disc_minibatch_size = 512  # Num discriminator updates per batch
    gen_batch_size = 2048  # Batch size for generator updates.


@train_ex.config
def aliases_default_gen_batch_size(gen_batch_size):
    # Setting generator buffer capacity and discriminator batch size to
    # the same number is equivalent to not using a replay buffer at all.
    # "Disabling" the replay buffer seems to improve convergence speed, but may
    # come at a cost of stability.
    gen_replay_buffer_size = gen_batch_size  # Num generator transitions stored


@train_ex.config
def apply_init_trainer_kwargs_aliases(
    disc_minibatch_size, disc_batch_size, gen_replay_buffer_size
):
    init_trainer_kwargs = dict(
        trainer_kwargs=dict(
            disc_minibatch_size=disc_minibatch_size,
            gen_replay_buffer_capacity=gen_replay_buffer_size,
            disc_batch_size=disc_batch_size,
        )
    )


@train_ex.config
def calc_n_steps(init_trainer_kwargs, gen_batch_size):
    _num_vec = init_trainer_kwargs["num_vec"]
    assert gen_batch_size % _num_vec == 0, (
        "num_vec must evenly divide " "gen_batch_size"
    )
    init_trainer_kwargs["init_rl_kwargs"]["n_steps"] = gen_batch_size // _num_vec
    del _num_vec


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

    assert os.path.exists(rollout_path), rollout_path


# Training algorithm named configs


@train_ex.named_config
def gail():
    init_trainer_kwargs = dict(use_gail=True,)


@train_ex.named_config
def airl():
    init_trainer_kwargs = dict(use_gail=False,)


# Shared settings

MUJOCO_SHARED_LOCALS = dict(init_trainer_kwargs=dict(airl_entropy_weight=0.1,),)

ANT_SHARED_LOCALS = dict(
    total_timesteps=8e7,
    gen_batch_size=2048 * 8,
    disc_batch_size=2048 * 8,
    init_trainer_kwargs=dict(max_episode_steps=500,),  # To match `inverse_rl` settings.
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
    init_trainer_kwargs = dict(scale=False,)
    total_timesteps = 3e7


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
    init_trainer_kwargs = dict(
        init_rl_kwargs=dict(policy_network_class=policies.MlpPolicy,),
    )


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
    total_timesteps = 10
    n_expert_demos = 1
    n_episodes_eval = 1
    gen_batch_size = 2
    disc_batch_size = 2
    disc_minibatch_size = 2
    show_plots = False
    n_plot_episodes = 1
    init_trainer_kwargs = dict(
        parallel=False,  # easier to debug with everything in one process
        max_episode_steps=1e2,
        num_vec=2,
        init_rl_kwargs=dict(nminibatches=1),
    )
