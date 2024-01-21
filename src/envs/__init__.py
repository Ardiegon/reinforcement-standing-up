from src.envs.env_mujoco import build_env as build_mujoco
from src.envs.env_pybullet import build_env as build_pybullet

def get_env(env_type):
    return {
        "mujoco":build_mujoco,
        "pybullet":build_pybullet
    }[env_type]()

__all__ = [
    get_env
]