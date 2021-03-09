"""Config files for parallel experiments.

Parallel experiments are intended to be defined in Python rather than
via CLI. For example, a user should add a new
`@parallel_ex.named_config` to define a new parallel experiment.

Adding custom named configs is necessary because the CLI interface can't add
search spaces to the config like `"seed": tune.grid_search([0, 1, 2, 3])`.
"""

import numpy as np
import ray.tune as tune
import sacred

from imitation.util.util import make_unique_timestamp

parallel_ex = sacred.Experiment("parallel")


@parallel_ex.config
def config():
    sacred_ex_name = "expert_demos"  # The experiment to parallelize
    init_kwargs = {"include_dashboard": False}  # Kwargs to pass to ray.init()
    _uuid = make_unique_timestamp()
    run_name = f"DEFAULT_{_uuid}"  # CLI --name option. For analysis grouping.
    resources_per_trial = {}  # Argument to `tune.run`
    base_named_configs = []  # Background settings before search_space is applied
    base_config_updates = {}  # Background settings before search_space is applied
    search_space = {
        "named_configs": [],
        "config_updates": {},
    }  # `config` argument to `ray.tune.run(trainable, config)`

    local_dir = None  # `local_dir` arg for `ray.tune.run`
    upload_dir = None  # `upload_dir` arg for `ray.tune.run`
    n_seeds = 3  # Number of seeds to search over by default


@parallel_ex.config
def seeds(n_seeds):
    search_space = {"config_updates": {"seed": tune.grid_search(list(range(n_seeds)))}}


@parallel_ex.named_config
def s3():
    upload_dir = "s3://shwang-chai/private"


# Debug named configs
@parallel_ex.named_config
def debug_log_root():
    search_space = {"config_updates": {"log_root": "/tmp/debug_parallel_ex"}}


@parallel_ex.named_config
def generate_test_data():
    """Used by tests/generate_test_data.sh to generate tests/data/gather_tb/.

    "tests/data/gather_tb/" should contain 4 Tensorboard run directories ("sb_tb/" and
    "tb/" for each of two trials in the search space below).
    """
    sacred_ex_name = "expert_demos"
    run_name = "TEST"
    n_seeds = 1
    search_space = {
        "config_updates": {
            "init_rl_kwargs": {
                "learning_rate": tune.grid_search([3e-4 * x for x in (1 / 3, 1 / 2)]),
            },
        }
    }
    base_named_configs = ["cartpole", "fast"]
    base_config_updates = {
        "init_tensorboard": True,
        "rollout_save_final": False,
    }


@parallel_ex.named_config
def example_cartpole_rl():
    sacred_ex_name = "expert_demos"
    run_name = "example-cartpole"
    n_seeds = 2
    search_space = {
        "config_updates": {
            "init_rl_kwargs": {
                "learning_rate": tune.grid_search(np.logspace(3e-6, 1e-1, num=3)),
                "nminibatches": tune.grid_search([16, 32, 64]),
            },
        }
    }
    base_named_configs = ["cartpole"]
    base_config_updates = {"init_tensorboard": True}
    resources_per_trial = dict(cpu=4)


EASY_ENVS = ["cartpole", "pendulum", "mountain_car"]


@parallel_ex.named_config
def example_rl_easy():
    sacred_ex_name = "expert_demos"
    run_name = "example-rl-easy"
    n_seeds = 2
    search_space = {
        "named_configs": tune.grid_search([[env] for env in EASY_ENVS]),
        "config_updates": {
            "init_rl_kwargs": {
                "learning_rate": tune.grid_search(np.logspace(3e-6, 1e-1, num=3)),
                "nminibatches": tune.grid_search([16, 32, 64]),
            },
        },
    }
    base_config_updates = {"init_tensorboard": True}
    resources_per_trial = dict(cpu=4)


@parallel_ex.named_config
def example_gail_easy():
    sacred_ex_name = "train_adversarial"
    run_name = "example-gail-easy"
    n_seeds = 1
    search_space = {
        "named_configs": tune.grid_search([[env] for env in EASY_ENVS]),
        "config_updates": {
            "init_trainer_kwargs": {
                "init_rl_kwargs": {
                    "learning_rate": tune.grid_search(np.logspace(3e-6, 1e-1, num=3)),
                    "nminibatches": tune.grid_search([16, 32, 64]),
                },
            },
        },
    }
    base_config_updates = {
        "init_tensorboard": True,
        "init_trainer_kwargs": {"use_gail": True},
    }
