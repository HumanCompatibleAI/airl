"""Test imitation.envs.*."""

import gym
import pytest

# TODO(sam): seals_test seems to require MuJoCo. It would be helpful to skip
# these tests if it is not available, like we do in test_scripts.py.
from seals.testing import envs as seals_test

# Unused imports is for side-effect of registering environments
from imitation.envs import examples  # noqa: F401
from imitation.testing import envs as imitation_test

ENV_NAMES = [
    env_spec.id
    for env_spec in gym.envs.registration.registry.all()
    if env_spec.id.startswith("imitation/")
]

DETERMINISTIC_ENVS = []


env = pytest.fixture(seals_test.make_env_fixture(skip_fn=pytest.skip))


@pytest.mark.parametrize("env_name", ENV_NAMES)
class TestEnvs:
    """Battery of simple tests for environments."""

    def test_seed(self, env, env_name):
        seals_test.test_seed(env, env_name, DETERMINISTIC_ENVS)

    def test_premature_step(self, env):
        """Test that you must call reset() before calling step()."""
        seals_test.test_premature_step(
            env,
            skip_fn=pytest.skip,
            raises_fn=pytest.raises,
        )

    def test_model_based(self, env):
        """Smoke test for each of the ModelBasedEnv methods with type checks."""
        if not hasattr(env, "state_space"):  # pragma: no cover
            pytest.skip("This test is only for subclasses of ModelBasedEnv.")

        imitation_test.test_model_based(env)

    def test_rollout_schema(self, env: gym.Env):
        """Tests if environments have correct types on `step()` and `reset()`."""
        seals_test.test_rollout_schema(env)

    def test_render(self, env: gym.Env):
        """Tests `render()` supports modes specified in environment metadata."""
        seals_test.test_render(env, raises_fn=pytest.raises)
