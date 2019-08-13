"""Smoke tests for CLI programs in imitation.scripts.*"""

import tempfile

from imitation.scripts.data_collect import data_collect_ex
from imitation.scripts.policy_eval import policy_eval_ex
from imitation.scripts.train import train_ex


def test_data_collect_main():
  """Smoke test for imitation.scripts.data_collect.rollouts_and_policy"""
  run = data_collect_ex.run(
      named_configs=['cartpole', 'fast'],
      # codecov does not like parallel
      config_updates={'parallel': False},
  )
  assert run.status == 'COMPLETED'


def test_data_collect_rollouts_from_policy():
  """Smoke test for imitation.scripts.data_collect.rollouts_from_policy"""
  with tempfile.TemporaryDirectory(prefix='imitation-data_collect-policy',
                                   ) as tmpdir:
    run = data_collect_ex.run(
        command_name="rollouts_from_policy",
        named_configs=['cartpole', 'fast'],
        config_updates=dict(
          parallel=False,  # codecov does not like parallel
          rollout_save_dir=tmpdir,
          policy_path="expert_models/PPO2_CartPole-v1_0",
        ))
  assert run.status == 'COMPLETED'


def test_policy_eval():
  """Smoke test for imitation.scripts.policy_eval"""
  config_updates = {
      'render': False,
      'log_root': 'output/tests/policy_eval',
  }
  run = policy_eval_ex.run(config_updates=config_updates,
                           named_configs=['fast'])
  assert run.status == 'COMPLETED'
  assert isinstance(run.result, dict)


def test_train():
  """Smoke test for imitation.scripts.train"""
  config_updates = {
      'init_trainer_kwargs': {
          # codecov does not like parallel
          'parallel': False,
          # Rollouts are small, decrease size of buffer to avoid warning
          'trainer_kwargs': {
              'n_disc_samples_per_buffer': 50,
          },
      },
      'log_root': 'output/tests/train',
      'rollout_glob': "tests/data/rollouts/CartPole*.pkl",
  }
  run = train_ex.run(
      named_configs=['cartpole', 'gail', 'fast'],
      config_updates=config_updates,
  )
  assert run.status == 'COMPLETED'
