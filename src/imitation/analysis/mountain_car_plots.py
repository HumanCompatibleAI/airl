"""Heatmaps and reward plotting code for debugging MountainCar."""

import pathlib
import pickle
from typing import Dict, List, Optional, Sequence, Union

import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common import vec_env

from imitation.data import rollout, types
from imitation.policies import serialize as policies_serialize
from imitation.rewards import common
from imitation.rewards import serialize as rewards_serialize

MC_POS_MIN, MC_POS_MAX = -1.2, 0.6
MC_VEL_MIN, MC_VEL_MAX = -0.07, 0.07
MC_GOAL_POS = 0.5
MC_NUM_ACTS = 3

ACT_NAMES = ["left", "neutral", "right"]


def _make_next_mc_obs(obs, acts) -> np.ndarray:
    """Utility for calculating the MountainCar-v0 next observation s'.

    Required for evaluating AIRL reward at arbitrary (s, a) points.
    """
    env = gym.make("MountainCar-v0")
    obs = np.array(obs)
    acts = np.array(acts)
    next_obs = []
    for ob, act in zip(obs, acts):
        assert ob.shape == (2,)
        env.reset()
        env.unwrapped.state = ob
        next_ob = env.step(act)[0]
        next_obs.append(next_ob)
    return np.array(next_obs)


def make_heatmap(
    act: int,
    reward_fn: common.RewardFn,
    n_pos_step: int = 18,
    n_vel_step: int = 14,
    mark_goal: bool = True,
    gen_trajs: Optional[List[types.Trajectory]] = None,
    exp_trajs: Optional[List[types.Trajectory]] = None,
    legend_on: bool = True,
    title: bool = None,
    heatmap: bool = True,
    filter_trans_by_act: bool = True,
) -> plt.Figure:
    """Make a MountainCar heatmap of rewards for a particular action.

    X axis is position. Y axis is velocity.

    Args:
        act: The MountainCar action number whose reward we are evaluating.
            Should be 0, 1, or 2.
        reward_fn: Reward function. Should accept unnormalized inputs.
        n_pos_step: The number of squares that the x axis of the heatmap is divided
            into.
        n_vel_step: The number of squares that the y axis of the heatmap is divided
            into.
        gen_trajs: A list of generator trajectories to
            scatterplot on top of the heatmap.
        exp_trajs: A list of exp trajectories to scatterplot on
            top of the heatmap.
        legend_on: Whether to plot the legend.
        title: Custom title.
        heatmap: Whether to plot the heatmap.
        filter_trans_by_act: If True, then filter out transitions from
            `gen_trajs` and `exp_trajs` that don't use action `act` before
            scatterplotting.
    """
    assert 0 <= act < MC_NUM_ACTS

    def convert_traj_to_coords_filtered(trajs: Sequence[types.Trajectory]):
        trans = rollout.flatten_trajectories(trajs)
        obs = trans.obs
        if filter_trans_by_act:
            obs = obs[trans.acts == act]
        return obs[:, 0], obs[:, 1]

    fig, ax = plt.subplots()
    if heatmap:
        pos_space = np.linspace(MC_POS_MIN, MC_POS_MAX, n_pos_step, endpoint=True)
        vel_space = np.linspace(MC_VEL_MIN, MC_VEL_MAX, n_vel_step, endpoint=True)
        obs_vec = np.array([[p, v] for p in pos_space for v in vel_space])
        acts_vec = np.array([act] * len(obs_vec))
        next_obs_vec = _make_next_mc_obs(obs_vec, acts_vec)
        dones = np.zeros(len(acts_vec), dtype=np.bool)

        rew = reward_fn(obs_vec, acts_vec, next_obs_vec, dones)
        # Transpose because `pcolor` (confusingly) expects its first two arguments
        # to be XY, but its matrix argument to be in RC format.
        rew_matrix = rew.reshape(n_pos_step, n_vel_step).T
        c = ax.pcolor(pos_space, vel_space, rew_matrix)
        fig.colorbar(c, ax=ax)

    if mark_goal:
        ax.axvline(
            x=MC_GOAL_POS, linestyle="--", label=f"goal state (pos={MC_GOAL_POS})"
        )
    if exp_trajs is not None:
        X, Y = convert_traj_to_coords_filtered(exp_trajs)
        ax.scatter(X, Y, marker="o", label="expert samples", alpha=0.2)
    if gen_trajs is not None:
        X, Y = convert_traj_to_coords_filtered(gen_trajs)
        ax.scatter(X, Y, marker="o", c="yellow", label="policy samples", alpha=0.2)

    if title is None:
        title = f"Action {ACT_NAMES[act]}"
    ax.set_title(title)
    if legend_on:
        ax.legend(loc="center left", bbox_to_anchor=(0, 1.3))

    return fig


def batch_reward_heatmaps(
    checkpoints_dir: Union[str, pathlib.Path],
    n_gen_trajs: int = 50,
    exp_trajs: Optional[List[types.Trajectory]] = None,
) -> Dict[pathlib.Path, plt.Figure]:
    """Build multiple mountain car reward heatmaps from a checkpoint directory.

    One plot is generated for every combination of action and checkpoint timestep.

    Args:
        checkpoints_dir: Path to `checkpoint/` directory from AIRL or GAIL output
            directory.
        n_gen_trajs: The number of trajectories to rollout using each generator
            checkpoint. The transitions in the trajectory are scatterplotted on top of
            the heatmap from the same checkpoint timestamp. Nonpositive indicates that
            no trajectories should be plotted.
        exp_trajs: Expert trajectories for scatterplotting. Generator trajectories
            are dynamically generated from generator checkpoints.

    Returns:
        A dictionary mapping relative paths to `plt.Figure`. Every key is of the
        form "{action_name}/{checkpoint_step}" where action_name is "left",
        "neutral", or "right".
    """
    result = {}
    venv = vec_env.DummyVecEnv([lambda: gym.make("MountainCar-v0")])
    checkpoints_dir = pathlib.Path(checkpoints_dir)
    for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
        vec_normalize_path = checkpoint_dir / "gen_policy" / "vec_normalize.pkl"
        discrim_path = checkpoint_dir / "discrim.pt"
        policy_path = checkpoint_dir / "gen_policy"

        if n_gen_trajs > 0:
            # `load_policy` automatically loads VecNormalize for policy evaluation.
            gen_policy = policies_serialize.load_policy("ppo", str(policy_path), venv)
            gen_trajs = rollout.generate_trajectories(
                gen_policy, venv, sample_until=rollout.min_episodes(n_gen_trajs)
            )
        else:
            gen_trajs = None

        # `gen_trajs` contains unnormalized observations.
        # Load VecNormalize for use in RewardFn, which doesn't automatically
        # normalize input observations.
        with open(vec_normalize_path, "rb") as f:
            vec_normalize = pickle.load(f)  # type: vec_env.VecNormalize
        vec_normalize.training = False

        reward_fn = rewards_serialize.load_reward("DiscrimNet", discrim_path, venv)
        norm_rew_fn = common.build_norm_reward_fn(
            reward_fn=reward_fn, vec_normalize=vec_normalize
        )
        for act in range(MC_NUM_ACTS):
            fig = make_heatmap(
                act=act,
                reward_fn=norm_rew_fn,
                gen_trajs=gen_trajs,
                exp_trajs=exp_trajs,
            )
            path = pathlib.Path(ACT_NAMES[act], checkpoint_dir.name)
            result[path] = fig
    return result


def plot_reward_vs_time(
    trajs_dict: Dict[str, List[types.Trajectory]],
    reward_fn: common.RewardFn,
    preferred_colors: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Plots a reward versus timestep line for each Trajectory.

    Args:
        trajs_dict: A dictionary mapping rollout labels (e.g. "expert" or
            "gen policy") to rollouts associated with those labels.
        reward_fn: Reward function for evaluating rollout rewards.
        preferred_colors: An optional dictionary mapping rollout labels to
            preferred line colors.

    Returns:
        The figure.
    """
    if preferred_colors is None:
        preferred_colors = {}
    fig, ax = plt.subplots()

    for i, (trajs_label, trajs_list) in enumerate(trajs_dict.items()):
        X = []
        Y = []
        for traj in trajs_list:
            T = len(traj.acts)
            X.extend(range(T))
            dones = np.zeros(T, dtype=bool)
            dones[-1] = True
            rews = reward_fn(traj.obs[:-1], traj.acts, traj.obs[1:], dones)
            Y.extend(rews)
        color = preferred_colors.get(trajs_label, None)
        ax.plot(X, Y, label=trajs_label, c=color)
    ax.set_xlabel("timestep")
    ax.set_ylabel("test reward")
    ax.legend()
    return fig
