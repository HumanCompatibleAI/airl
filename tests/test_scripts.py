"""Smoke tests for CLI programs in imitation.scripts.*

Every test in this file should use `parallel=False` to turn off multiprocessing because
codecov might interact poorly with multiprocessing. The 'fast' named_config for each
experiment implicitly sets parallel=False.
"""

import os.path as osp
import tempfile
from collections import Counter
from typing import List, Optional

import pandas as pd
import pytest
import ray.tune as tune

from imitation.scripts.analyze import analysis_ex
from imitation.scripts.eval_policy import eval_policy_ex
from imitation.scripts.expert_demos import expert_demos_ex
from imitation.scripts.parallel import parallel_ex
from imitation.scripts.train_adversarial import train_ex


def test_expert_demos_main(tmpdir):
    """Smoke test for imitation.scripts.expert_demos.rollouts_and_policy."""
    run = expert_demos_ex.run(
        named_configs=["cartpole", "fast"], config_updates=dict(log_root=tmpdir,),
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)


def test_expert_demos_rollouts_from_policy(tmpdir):
    """Smoke test for imitation.scripts.expert_demos.rollouts_from_policy."""
    run = expert_demos_ex.run(
        command_name="rollouts_from_policy",
        named_configs=["cartpole", "fast"],
        config_updates=dict(
            log_root=tmpdir,
            rollout_save_path=osp.join(tmpdir, "rollouts", "test.pkl"),
            policy_path="tests/data/expert_models/cartpole_0/policies/final/",
        ),
    )
    assert run.status == "COMPLETED"


EVAL_POLICY_CONFIGS = [
    {},
    {"reward_type": "zero", "reward_path": "foobar"},
]


@pytest.mark.parametrize("config", EVAL_POLICY_CONFIGS)
def test_eval_policy(config, tmpdir):
    """Smoke test for imitation.scripts.eval_policy."""
    config_updates = {
        "render": False,
        "log_root": tmpdir,
    }
    config_updates.update(config)
    run = eval_policy_ex.run(config_updates=config_updates, named_configs=["fast"])
    assert run.status == "COMPLETED"
    wrapped_reward = "reward_type" in config
    _check_rollout_stats(run.result, wrapped_reward)


def _check_rollout_stats(stats: dict, wrapped_reward: bool = True):
    """Common assertions for rollout_stats."""
    assert isinstance(stats, dict)
    assert "return_mean" in stats
    assert "monitor_return_mean" in stats
    if wrapped_reward:
        # If the reward is wrapped, then we expect the monitor return
        # to differ.
        assert stats.get("return_mean") != stats.get("monitor_return_mean")
    else:
        assert stats.get("return_mean") == stats.get("monitor_return_mean")


def _check_train_ex_result(result: dict):
    expert_stats = result.get("expert_stats")
    assert isinstance(expert_stats, dict)
    assert "return_mean" in expert_stats
    assert "monitor_return_mean" not in expert_stats

    imit_stats = result.get("imit_stats")
    _check_rollout_stats(imit_stats)


def test_train_adversarial(tmpdir):
    """Smoke test for imitation.scripts.train_adversarial."""
    named_configs = ["cartpole", "gail", "fast"]
    config_updates = {
        "log_root": tmpdir,
        "rollout_path": "tests/data/expert_models/cartpole_0/rollouts/final.pkl",
        "init_tensorboard": True,
        "plot_interval": 1,
        "extra_episode_data_interval": 1,
    }
    run = train_ex.run(named_configs=named_configs, config_updates=config_updates,)
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)


def test_transfer_learning(tmpdir):
    """Transfer learning smoke test.

    Saves a dummy AIRL test reward, then loads it for transfer learning.
    """
    log_dir_train = osp.join(tmpdir, "train")
    run = train_ex.run(
        named_configs=["cartpole", "airl", "fast"],
        config_updates=dict(
            rollout_path="tests/data/expert_models/cartpole_0/rollouts/final.pkl",
            log_dir=log_dir_train,
        ),
    )
    assert run.status == "COMPLETED"
    _check_train_ex_result(run.result)

    _check_rollout_stats(run.result["imit_stats"])

    log_dir_data = osp.join(tmpdir, "expert_demos")
    discrim_path = osp.join(log_dir_train, "checkpoints", "final", "discrim")
    run = expert_demos_ex.run(
        named_configs=["cartpole", "fast"],
        config_updates=dict(
            log_dir=log_dir_data, reward_type="DiscrimNet", reward_path=discrim_path,
        ),
    )
    assert run.status == "COMPLETED"
    _check_rollout_stats(run.result)


PARALLEL_CONFIG_UPDATES = [
    dict(
        sacred_ex_name="expert_demos",
        base_named_configs=["cartpole", "fast"],
        n_seeds=2,
        search_space={
            "config_updates": {
                "init_rl_kwargs": {"learning_rate": tune.grid_search([3e-4, 1e-4])},
            }
        },
    ),
    dict(
        sacred_ex_name="train_adversarial",
        base_named_configs=["cartpole", "gail", "fast"],
        base_config_updates={
            # Need absolute path because raylet runs in different working directory.
            "rollout_path": osp.abspath(
                "tests/data/expert_models/cartpole_0/rollouts/final.pkl"
            ),
        },
        search_space={
            "config_updates": {
                "init_trainer_kwargs": {
                    "reward_kwargs": {
                        "phi_units": tune.grid_search([[16, 16], [7, 9]]),
                    },
                },
            }
        },
    ),
]

PARALLEL_CONFIG_LOW_RESOURCE = {
    # CI server only has 2 cores.
    "init_kwargs": {"num_cpus": 2},
    # Memory is low enough we only want to run one job at a time.
    "resources_per_trial": {"cpu": 2},
}


@pytest.mark.parametrize("config_updates", PARALLEL_CONFIG_UPDATES)
def test_parallel(config_updates):
    """Hyperparam tuning smoke test."""
    # CI server only has 2 cores
    config_updates = dict(config_updates)
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    # No need for TemporaryDirectory because the hyperparameter tuning script
    # itself generates no artifacts, and "debug_log_root" sets inner experiment's
    # log_root="/tmp/parallel_debug/".
    run = parallel_ex.run(
        named_configs=["debug_log_root"], config_updates=config_updates
    )
    assert run.status == "COMPLETED"


def _generate_test_rollouts(tmpdir: str, env_named_config: str) -> str:
    expert_demos_ex.run(
        named_configs=[env_named_config, "fast"],
        config_updates=dict(rollout_save_interval=0, log_dir=tmpdir,),
    )
    rollout_path = osp.abspath(f"{tmpdir}/rollouts/final.pkl")
    return rollout_path


def test_parallel_train_adversarial_custom_env(tmpdir):
    env_named_config = "custom_ant"
    rollout_path = _generate_test_rollouts(tmpdir, env_named_config)

    config_updates = dict(
        sacred_ex_name="train_adversarial",
        n_seeds=1,
        base_named_configs=[env_named_config, "fast"],
        base_config_updates=dict(
            init_trainer_kwargs=dict(parallel=True, num_vec=2,),
            rollout_path=rollout_path,
        ),
    )
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    run = parallel_ex.run(
        named_configs=["debug_log_root"], config_updates=config_updates
    )
    assert run.status == "COMPLETED"


@pytest.mark.parametrize("run_names", ([], list("adab")))
def test_analyze_imitation(tmpdir: str, run_names: List[str]):
    sacred_logs_dir = tmpdir

    # Generate sacred logs (other logs are put in separate tmpdir for deletion).
    for i, run_name in enumerate(run_names):
        with tempfile.TemporaryDirectory(prefix="junk") as junkdir:
            rollout_path = "tests/data/expert_models/cartpole_0/rollouts/final.pkl"
            run = train_ex.run(
                named_configs=["cartpole", "fast"],
                config_updates=dict(
                    rollout_path=rollout_path, log_dir=junkdir, checkpoint_interval=-1,
                ),
                options={"--name": run_name, "--file_storage": sacred_logs_dir},
            )
            assert run.status == "COMPLETED"

    # Check that analyze script finds the correct number of logs.
    def check(run_name: Optional[str], count: int) -> None:
        run = analysis_ex.run(
            command_name="analyze_imitation",
            config_updates=dict(
                source_dir=sacred_logs_dir,
                run_name=run_name,
                csv_output_path=osp.join(tmpdir, "analysis.csv"),
                verbose=True,
            ),
        )
        assert run.status == "COMPLETED"
        df = pd.DataFrame(run.result)
        assert df.shape[0] == count

    for run_name, count in Counter(run_names).items():
        check(run_name, count)

    check(None, len(run_names))  # Check total number of logs.


def test_analyze_gather_tb(tmpdir: str):
    config_updates = dict(local_dir=tmpdir, run_name="test")
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    parallel_run = parallel_ex.run(
        named_configs=["generate_test_data"], config_updates=config_updates
    )
    assert parallel_run.status == "COMPLETED"

    run = analysis_ex.run(
        command_name="gather_tb_directories", config_updates=dict(source_dir=tmpdir,)
    )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, dict)
    assert run.result["n_tb_dirs"] == 4
