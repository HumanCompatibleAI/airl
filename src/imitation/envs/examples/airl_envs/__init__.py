from typing import Optional

from gym.envs import register as gym_register

_ENTRY_POINT_PREFIX = "imitation.envs.examples.airl_envs"


def _register(env_name: str, entry_point: str, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    gym_register(id=env_name, entry_point=entry_point, kwargs=kwargs)


def _point_maze_register():
    for dname, dval in {"Left": 0, "Right": 1}.items():
        for vname, vval in {"": False, "Vel": True}.items():
            _register(
                f"imitation/PointMaze{dname}{vname}-v0",
                entry_point="point_maze_env:PointMazeEnv",
                kwargs={"direction": dval, "include_vel": vval},
            )


_register(
    "imitation/ObjPusher-v0",
    entry_point="pusher_env:PusherEnv",
    kwargs={"sparse_reward": False},
)
_register("imitation/TwoDMaze-v0", entry_point="twod_maze:TwoDMaze")

_point_maze_register()

# A modified ant which flips over less and learns faster via TRPO
_register(
    "imitation/CustomAnt-v0",
    entry_point="ant_env:CustomAntEnv",
    kwargs={"gear": 30, "disabled": False},
)
_register(
    "imitation/DisabledAnt-v0",
    entry_point="ant_env:CustomAntEnv",
    kwargs={"gear": 30, "disabled": True},
)
