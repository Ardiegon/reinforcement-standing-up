import gym.envs.registration
import numpy as np
from gym.utils.env_checker import check_env

# Fix gym and PyBullet compatibility
from collections import UserDict

from gym.wrappers import ResizeObservation

# registry = UserDict(gym.envs.registration.registry)
# registry.env_specs = gym.envs.registration.registry
# gym.envs.registration.registry = registry

from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv


class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env, new_step_api=True)

    def observation(self, observation):
        new_observation = np.concatenate([observation, np.zeros(332)])
        return new_observation, {}

    def step(self, action):
        obs, reward, terminated, truncated = super().step(action)
        return (obs[0], reward, terminated, truncated, obs[1])


def build_env():
    env = TransformObservation(env=HumanoidBulletEnv(render=True))
    # check_env(env)
    # env.seed(42)
    # env.action_space.seed(42)
    # env.reset()
    return env

