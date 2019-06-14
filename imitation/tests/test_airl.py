import gin.tf
import pytest
import tensorflow as tf

import imitation.util as util
from imitation.util.trainer import init_trainer

use_gail_vals = [True, False]
gin.parse_config_file("configs/test.gin")
tf.logging.set_verbosity(tf.logging.INFO)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    yield
    tf.reset_default_graph()


@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_init_no_crash(use_gail, env='CartPole-v1'):
    init_trainer(env, use_gail=use_gail)


@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_train_disc_no_crash(use_gail, env='CartPole-v1', n_timesteps=200):
    trainer = init_trainer(env, use_gail=use_gail)
    trainer.train_disc()
    obs_old, act, obs_new, _ = util.rollout.generate(
            trainer.gen_policy, env, n_timesteps=n_timesteps)
    trainer.train_disc(gen_old_obs=obs_old, gen_act=act,
                       gen_new_obs=obs_new)


@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_train_gen_no_crash(use_gail, env='CartPole-v1', n_steps=10):
    trainer = init_trainer(env, use_gail=use_gail)
    trainer.train_gen(n_steps)


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_train_disc_improve_D(use_gail, env='CartPole-v1', n_timesteps=200,
                              n_steps=1000):
    trainer = init_trainer(env, use_gail=use_gail)
    obs_old, act, obs_new, _ = util.rollout.generate(
            trainer.gen_policy, env, n_timesteps=n_timesteps)
    kwargs = dict(gen_old_obs=obs_old, gen_act=act, gen_new_obs=obs_new)
    loss1 = trainer.eval_disc_loss(**kwargs)
    trainer.train_disc(n_steps=n_steps, **kwargs)
    loss2 = trainer.eval_disc_loss(**kwargs)
    assert loss2 < loss1


# Not testing with GAIL because it's flaky.
@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", [False])
def test_train_gen_degrade_D(use_gail, env='CartPole-v1', n_timesteps=200,
                             n_steps=10000):
    trainer = init_trainer(env, use_gail=use_gail)
    if use_gail:
        kwargs = {}
    else:
        obs_old, act, obs_new, _ = util.rollout.generate(
                trainer.gen_policy, env, n_timesteps=n_timesteps)
        kwargs = dict(gen_old_obs=obs_old, gen_act=act, gen_new_obs=obs_new)

    loss1 = trainer.eval_disc_loss(**kwargs)
    trainer.train_gen(n_steps=n_steps)
    loss2 = trainer.eval_disc_loss(**kwargs)
    assert loss2 > loss1


# Not testing with GAIL because it's flaky.
@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", [False])
def test_train_disc_then_gen(use_gail, env='CartPole-v1', n_timesteps=200,
                             n_steps=10000):
    trainer = init_trainer(env, use_gail=use_gail)
    if use_gail:
        kwargs = {}
    else:
        obs_old, act, obs_new, _ = util.rollout.generate(
            trainer.gen_policy, env, n_timesteps=n_timesteps)
        kwargs = dict(gen_old_obs=obs_old, gen_act=act, gen_new_obs=obs_new)

    loss1 = trainer.eval_disc_loss(**kwargs)
    trainer.train_disc(n_steps=n_steps, **kwargs)
    loss2 = trainer.eval_disc_loss(**kwargs)
    trainer.train_gen(n_steps=n_steps)
    loss3 = trainer.eval_disc_loss(**kwargs)
    assert loss2 < loss1
    assert loss3 > loss2


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_train_no_crash(use_gail, env='CartPole-v1'):
    trainer = init_trainer(env, use_gail=use_gail)
    trainer.train(n_epochs=1)


@pytest.mark.expensive
@pytest.mark.xfail(
        reason="Either AIRL train is broken or not enough epochs."
        " Consider making a plot of episode reward over time to check.")
@pytest.mark.skip
def test_trained_policy_better_than_random(use_gail, env='CartPole-v1',
                                           n_episodes=50):
    """
    Make sure that generator policy trained to mimick expert policy
    demonstrations) achieves higher reward than a random policy.

    In other words, perform a basic check on the imitation learning
    capabilities of AIRLTrainer.
    """
    env = util.make_vec_env(env, 32)
    trainer = init_trainer(env, use_expert_rollouts=True, use_gail=use_gail)
    expert_policy = util.load_policy(env, basedir="expert_models")
    random_policy = util.make_blank_policy(env)
    if expert_policy is None:
        pytest.fail("Couldn't load expert_policy!")

    trainer.train(n_epochs=200)

    # Idea: Plot n_epochs vs generator reward.
    for _ in range(4):
        expert_rew = util.rollout.total_reward(expert_policy, env,
                                               n_episodes=n_episodes)
        gen_rew = util.rollout.total_reward(trainer.gen_policy, env,
                                            n_episodes=n_episodes)
        random_rew = util.rollout.total_reward(random_policy, env,
                                               n_episodes=n_episodes)

        print("expert reward:", expert_rew)
        print("generator reward:", gen_rew)
        print("random reward:", random_rew)
        assert expert_rew > random_rew
        assert gen_rew > random_rew


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_wrap_learned_reward_no_crash(use_gail, env="CartPole-v1"):
    """
    Briefly train with AIRL, and then used the learned reward to wrap
    a duplicate environment. Finally, use that learned reward to train
    a policy.
    """
    trainer = init_trainer(env, use_gail=use_gail)
    trainer.train(n_epochs=1)

    learned_reward_env = trainer.wrap_env_test_reward(env)
    policy = util.make_blank_policy(env, init_tensorboard=False)
    policy.set_env(learned_reward_env)
    policy.learn(10)
