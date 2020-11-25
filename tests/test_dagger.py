"""Tests for DAgger."""
import glob
import os

import gym
import numpy as np
import pytest
from stable_baselines3.common import policies

from imitation.algorithms import bc, dagger
from imitation.data import rollout
from imitation.policies import serialize
from imitation.util import util

ENV_NAME = "CartPole-v1"
EXPERT_POLICY_PATH = "tests/data/expert_models/cartpole_0/policies/final/"


def test_beta_schedule():
    one_step_sched = dagger.LinearBetaSchedule(1)
    three_step_sched = dagger.LinearBetaSchedule(3)
    for i in range(10):
        assert np.allclose(one_step_sched(i), 1 if i == 0 else 0)
        assert np.allclose(three_step_sched(i), (3 - i) / 3 if i <= 2 else 0)


def test_traj_collector(tmpdir):
    env = gym.make(ENV_NAME)
    robot_calls = 0

    def get_random_act(obs):
        nonlocal robot_calls
        robot_calls += 1
        return env.action_space.sample()

    collector = dagger.InteractiveTrajectoryCollector(
        env=env, get_robot_act=get_random_act, beta=0.5, save_dir=tmpdir
    )
    collector.reset()
    zero_action = np.zeros((), dtype="int")
    obs, rew, done, info = collector.step(zero_action)
    assert rew != 0
    assert not done
    assert isinstance(info, dict)
    # roll out ~5 episodes
    for i in range(999):
        _, _, done, _ = collector.step(zero_action)
        if done:
            collector.reset()

    # there is a <10^(-12) probability this fails by chance; we should be calling
    # robot with 50% prob each time
    assert 388 <= robot_calls <= 612

    file_paths = glob.glob(os.path.join(tmpdir, "dagger-demo-*.npz"))
    assert len(file_paths) >= 5
    trajs = map(dagger._load_trajectory, file_paths)
    nonzero_acts = sum(np.sum(traj.acts != 0) for traj in trajs)
    assert nonzero_acts == 0


def make_trainer(tmpdir, beta_schedule=dagger.LinearBetaSchedule(1)):
    env = gym.make(ENV_NAME)
    env.seed(42)
    return dagger.DAggerTrainer(
        env,
        tmpdir,
        beta_schedule,
    )


@pytest.fixture(params=[None, dagger.LinearBetaSchedule(1)])
def trainer(request, tmpdir):
    beta_sched = request.param
    return make_trainer(tmpdir, beta_sched)


def test_trainer_makes_progress(trainer):
    venv = util.make_vec_env(ENV_NAME, 10)
    with pytest.raises(dagger.NeedsDemosException):
        trainer.extend_and_update()
    assert trainer.round_num == 0
    pre_train_rew_mean = rollout.mean_return(
        trainer.bc_trainer.policy,
        venv,
        sample_until=rollout.min_episodes(15),
        deterministic_policy=False,
    )
    # checking that the initial policy is poor can be flaky; sometimes the
    # randomly initialised policy performs very well, and it's not clear why
    # assert pre_train_rew_mean < 100
    expert_policy = serialize.load_policy("ppo", EXPERT_POLICY_PATH, venv)
    for i in range(2):
        # roll out a few trajectories for dataset, then train for a few steps
        collector = trainer.get_trajectory_collector()
        for _ in range(5):
            obs = collector.reset()
            done = False
            while not done:
                (expert_action,), _ = expert_policy.predict(
                    obs[None], deterministic=True
                )
                obs, _, done, _ = collector.step(expert_action)
        trainer.extend_and_update(n_epochs=1)
    # make sure we're doing better than a random policy would
    post_train_rew_mean = rollout.mean_return(
        trainer.bc_trainer.policy,
        venv,
        sample_until=rollout.min_episodes(15),
        deterministic_policy=True,
    )
    assert post_train_rew_mean - pre_train_rew_mean > 50, (
        f"pre-train mean {pre_train_rew_mean}, post-train mean "
        f"{post_train_rew_mean}"
    )


def test_trainer_save_reload(tmpdir, trainer):
    trainer.round_num = 3
    trainer.save_trainer()
    new_trainer = dagger.reconstruct_trainer(tmpdir)
    assert new_trainer.round_num == trainer.round_num

    # old trainer and reloaded trainer should have same variable values
    old_vars = trainer.bc_trainer.policy.state_dict()
    new_vars = new_trainer.bc_trainer.policy.state_dict()
    assert len(new_vars) == len(old_vars)
    for var, values in new_vars.items():
        assert values.equal(old_vars[var])

    # also those values should be different from a newly created trainer
    third_trainer = make_trainer(tmpdir)
    third_vars = third_trainer.bc_trainer.policy.state_dict()
    assert len(third_vars) == len(old_vars)
    assert not all(values.equal(old_vars[var]) for var, values in third_vars.items())


def test_policy_save_reload(tmpdir):
    # just make sure the methods run; we already test them in test_bc.py
    policy_path = os.path.join(tmpdir, "policy.pt")
    trainer = make_trainer(tmpdir)
    trainer.save_policy(policy_path)
    pol = bc.reconstruct_policy(policy_path)
    assert isinstance(pol, policies.BasePolicy)
