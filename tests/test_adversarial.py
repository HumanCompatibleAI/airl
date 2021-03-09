"""Tests for imitation.trainer.Trainer and util.trainer.init_trainer."""

import os

import pytest
import stable_baselines3.common.vec_env as sb3_vec_env
from torch.utils import data as th_data

import imitation.policies.base as pol_base
from imitation.algorithms import adversarial
from imitation.data import rollout, types
from imitation.testing import counter, normalization
from imitation.util import logger, util

ALGORITHM_CLS = [adversarial.AIRL, adversarial.GAIL]
IN_CODECOV = "COV_CORE_CONFIG" in os.environ
# Disable SubprocVecEnv tests for code coverage test since
# multiprocessing support is flaky in py.test --cov
PARALLEL = [False] if IN_CODECOV else [True, False]
CARTPOLE_PATH = "tests/data/expert_models/cartpole_0/rollouts/final.pkl"


@pytest.fixture(params=ALGORITHM_CLS)
def _algorithm_cls(request):
    """Auto-parametrizes `_algorithm_cls` for the `trainer` fixture."""
    return request.param


def test_train_disc_small_expert_data_warning(tmpdir, _algorithm_cls):
    logger.configure(tmpdir, ["tensorboard", "stdout"])
    venv = util.make_vec_env(
        "CartPole-v1",
        n_envs=2,
        parallel=_parallel,
    )

    gen_algo = util.init_rl(venv, verbose=1)
    small_data = rollout.generate_transitions(gen_algo, venv, n_timesteps=20)

    with pytest.raises(ValueError, match="Transitions.*expert_batch_size"):
        _algorithm_cls(
            venv=venv,
            expert_data=small_data,
            expert_batch_size=21,
            gen_algo=gen_algo,
            log_dir=tmpdir,
        )

    with pytest.raises(ValueError, match="expert_batch_size.*positive"):
        _algorithm_cls(
            venv=venv,
            expert_data=small_data,
            expert_batch_size=-1,
            gen_algo=gen_algo,
            log_dir=tmpdir,
        )


@pytest.fixture(params=PARALLEL)
def _parallel(request):
    """Auto-parametrizes `_parallel` for the `trainer` fixture.

    This way we don't have to add a @pytest.mark.parametrize("_parallel", ... )
    decorator in front of every test. I couldn't find a better way to do this that
    didn't involve the aforementioned `parameterize` duplication."""
    return request.param


@pytest.fixture(params=[True, False])
def _convert_dataset(request):
    """Auto-parametrizes `_convert_dataset` for the `trainer` fixture."""
    return request.param


@pytest.fixture(params=[1, 128])
def expert_batch_size(request):
    return request.param


@pytest.fixture
def expert_transitions():
    trajs = types.load("tests/data/expert_models/cartpole_0/rollouts/final.pkl")
    trans = rollout.flatten_trajectories(trajs)
    return trans


@pytest.fixture
def trainer(
    _algorithm_cls,
    _parallel: bool,
    tmpdir: str,
    _convert_dataset: bool,
    expert_batch_size: int,
    expert_transitions: types.Transitions,
):
    logger.configure(tmpdir, ["tensorboard", "stdout"])
    if _convert_dataset:
        expert_data = th_data.DataLoader(
            expert_transitions,
            batch_size=expert_batch_size,
            collate_fn=types.transitions_collate_fn,
            shuffle=True,
            drop_last=True,
        )
    else:
        expert_data = expert_transitions

    venv = util.make_vec_env(
        "CartPole-v1",
        n_envs=2,
        parallel=_parallel,
        log_dir=tmpdir,
    )

    gen_algo = util.init_rl(venv, verbose=1)

    trainer = _algorithm_cls(
        venv=venv,
        expert_data=expert_data,
        expert_batch_size=expert_batch_size,
        gen_algo=gen_algo,
        log_dir=tmpdir,
    )

    try:
        yield trainer
    finally:
        venv.close()


def test_train_disc_no_samples_error(trainer: adversarial.AdversarialTrainer):
    with pytest.raises(RuntimeError, match="No generator samples"):
        trainer.train_disc()


def test_train_disc_unequal_expert_gen_samples_error(trainer, expert_transitions):
    """Test that train_disc raises error when n_gen != n_expert samples."""
    if len(expert_transitions) < 2:  # pragma: no cover
        raise ValueError("Test assumes at least 2 samples.")
    expert_samples = types.dataclass_quick_asdict(expert_transitions)
    gen_samples = types.dataclass_quick_asdict(expert_transitions[:-1])
    with pytest.raises(ValueError, match="n_expert"):
        trainer.train_disc(expert_samples=expert_samples, gen_samples=gen_samples)


def test_train_disc_step_no_crash(trainer, expert_batch_size):
    transitions = rollout.generate_transitions(
        trainer.gen_algo,
        trainer.venv,
        n_timesteps=expert_batch_size,
        truncate=True,
    )
    trainer.train_disc(gen_samples=types.dataclass_quick_asdict(transitions))


def test_train_gen_train_disc_no_crash(trainer, n_updates=2):
    trainer.train_gen(n_updates * trainer.gen_batch_size)
    trainer.train_disc()


@pytest.mark.expensive
def test_train_disc_improve_D(
    tmpdir, trainer, expert_transitions, expert_batch_size, n_steps=3
):
    expert_samples = expert_transitions[:expert_batch_size]
    expert_samples = types.dataclass_quick_asdict(expert_samples)
    gen_samples = rollout.generate_transitions(
        trainer.gen_algo,
        trainer.venv_train,
        n_timesteps=expert_batch_size,
        truncate=True,
    )
    gen_samples = types.dataclass_quick_asdict(gen_samples)
    init_stats = final_stats = None
    for _ in range(n_steps):
        final_stats = trainer.train_disc(
            gen_samples=gen_samples, expert_samples=expert_samples
        )
        if init_stats is None:
            init_stats = final_stats
    assert final_stats["disc_loss"] < init_stats["disc_loss"]


def test_augment(_algorithm_cls, tmpdir: str):
    logger.configure(tmpdir, ["tensorboard", "stdout"])
    trajs = types.load(CARTPOLE_PATH)
    expert_data = rollout.flatten_trajectories(trajs)
    venv = util.make_vec_env(
        "CartPole-v1",
        n_envs=2,
        parallel=False,
        log_dir=tmpdir,
    )
    gen_algo = util.init_rl(venv, verbose=1)
    mock_augment = counter.IdentityCounter()
    trainer = _algorithm_cls(
        venv=venv,
        expert_data=expert_data,
        gen_algo=gen_algo,
        log_dir=tmpdir,
        disc_augmentation_fn=mock_augment,
        expert_batch_size=32,
    )
    transitions = rollout.generate_transitions(
        trainer.gen_algo,
        trainer.venv,
        n_timesteps=32,
    )

    # make sure augmenter gets called
    assert mock_augment.ncalls == 0
    trainer.train_disc(
        gen_samples={
            "obs": transitions.obs,
            "next_obs": transitions.next_obs,
            "acts": transitions.acts,
            "dones": transitions.dones,
        }
    )
    assert mock_augment.ncalls > 0


def test_adversarial_normalization(_algorithm_cls, tmpdir: str):
    logger.configure(tmpdir, ["tensorboard", "stdout"])
    venv = sb3_vec_env.DummyVecEnv([normalization.NormalizationTestEnv] * 2)
    rand_pol = pol_base.RandomPolicy(venv.observation_space, venv.action_space)
    expert_transitions = rollout.generate_transitions(rand_pol, venv, n_timesteps=32)
    gen_algo = util.init_rl(
        venv,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=normalization.NormalizationTestFeatEx,
            features_extractor_kwargs=dict(norm_env=venv.envs[0], features_dim=5),
        ),
    )

    if _algorithm_cls is adversarial.GAIL:
        trainer = _algorithm_cls(
            venv=venv,
            expert_data=expert_transitions,
            gen_algo=gen_algo,
            log_dir=tmpdir,
            expert_batch_size=32,
            normalize_obs=False,
            discrim_kwargs=dict(
                discrim_net=normalization.NormalizationTestDiscriminator(
                    action_space=venv.action_space,
                    observation_space=venv.observation_space,
                    norm_env=venv.envs[0],
                    hid_sizes=(5, 7),
                )
            ),
        )
    elif _algorithm_cls is adversarial.AIRL:
        trainer = _algorithm_cls(
            venv=venv,
            expert_data=expert_transitions,
            gen_algo=gen_algo,
            log_dir=tmpdir,
            expert_batch_size=32,
            normalize_obs=False,
            reward_net_kwargs=dict(
                base_reward_net=normalization.NormalizationTestRewardMLP(
                    action_space=venv.action_space,
                    observation_space=venv.observation_space,
                    norm_env=venv.envs[0],
                    hid_sizes=(5, 7),
                    use_state=True,
                    use_action=True,
                    use_next_state=True,
                    use_done=True,
                ),
                potential_net=normalization.NormalizationTestRewardMLP(
                    action_space=venv.action_space,
                    observation_space=venv.observation_space,
                    norm_env=venv.envs[0],
                    hid_sizes=(5, 7),
                    use_state=True,
                    use_action=False,
                    use_next_state=False,
                    use_done=False,
                ),
            ),
        )
    else:
        raise TypeError(f"don't know how to handle '{_algorithm_cls}'")

    # these steps will fail with an assertion error if normalization is not
    # being done correctly (e.g. if it's doing double-normalization, or doing
    # standardization instead of just dividing by 255)
    trainer.train_gen(trainer.gen_batch_size)
    trainer.train_disc()

    # Verify that:
    # 1. Policy feature extractor actually gets called.
    # 2. Discriminator feature extractor gets called in GAIL.
    # 3. Base and potential net feature extractors actually get called in AIRL.
    # (these ensure that our assertions are working)
    assert trainer.gen_algo.policy.features_extractor.assert_calls > 0
    if _algorithm_cls is adversarial.GAIL:
        assert trainer.discrim.discriminator.assert_calls > 0
    elif _algorithm_cls is adversarial.AIRL:
        assert trainer.discrim.reward_net._base_reward_net.assert_calls > 0
        assert trainer.discrim.reward_net._potential_net.assert_calls > 0
    else:
        raise TypeError(f"don't know how to handle '{_algorithm_cls}'")
