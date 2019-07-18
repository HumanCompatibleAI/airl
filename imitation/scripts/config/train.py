import sacred

from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS
from imitation.util import FeedForward64Policy

train_ex = sacred.Experiment("train", interactive=True)


@train_ex.config
def train_defaults():
    n_epochs = 50
    n_disc_steps_per_epoch = 50
    n_gen_steps_per_epoch = 2048

    init_trainer_kwargs = dict(
        rollouts_dir="data/rollouts",
        n_rollout_dumps=1,
        num_vec=8,  # NOTE: changing this also changes the effective n_steps!
        reward_kwargs=dict(
            theta_units=[32, 32],
            phi_units=[32, 32],
        ),

        trainer_kwargs=dict(
            n_disc_samples_per_buffer=1000,
            # Setting buffer capacity and disc samples to 1000 effectively
            # disables the replay buffer. This seems to improve convergence
            # speed, but may come at a cost of stability.
            gen_replay_buffer_capacity=1000,
        ),

        # Some environments (e.g. CartPole) have float max as limits, which
        # breaks the scaling.
        discrim_scale=False,

        make_blank_policy_kwargs=DEFAULT_BLANK_POLICY_KWARGS,
    )


@train_ex.named_config
def gail():
    init_trainer_kwargs = dict(
        use_gail=True,
    )


@train_ex.named_config
def airl():
    init_trainer_kwargs = dict(
        use_gail=False,
    )


@train_ex.named_config
def ant():
    env_name = "Ant-v2"
    n_epochs = 2000


@train_ex.named_config
def cartpole():
    env_name = "CartPole-v1"


@train_ex.named_config
def halfcheetah():
    env_name = "HalfCheetah-v2"
    n_epochs = 1000

    init_trainer_kwargs = dict(
        discrim_kwargs=dict(entropy_weight=0.1),
    )


@train_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"


@train_ex.named_config
def swimmer():
    env_name = "Swimmer-v2"
    n_epochs = 1000
    init_trainer_kwargs = dict(
        make_blank_policy_kwargs=dict(
            policy_network_class=FeedForward64Policy,
        ),
    )


@train_ex.named_config
def fast():
    """Minimize the amount of computation. Useful for test cases."""
    n_epochs = 1
    interactive = False
    n_disc_steps_per_epoch = 1
    n_gen_steps_per_epoch = 1
    n_episodes_per_reward_data = 1
