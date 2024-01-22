import gym.envs.registration
import numpy as np

# Fix gym and PyBullet compatibility
from collections import UserDict

from gym.wrappers import ResizeObservation

# registry = UserDict(gym.envs.registration.registry)
# registry.env_specs = gym.envs.registration.registry
# gym.envs.registration.registry = registry

from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv

import time


class TransformObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, fps = 30, verbose = False, termination_on_fall = False, steps_to_reset = 1000):
        super().__init__(env, new_step_api=True)
        self.fps = fps
        self.verbose = verbose
        self.termination_on_fall = termination_on_fall
        self.steps = 0
        self.steps_to_reset = steps_to_reset

    def observation(self, observation):
        new_observation = np.concatenate([observation, np.zeros(332)])
        return new_observation, {}

    def step(self, action):
        obs, reward, terminated, truncated = super().step(action)
        self.steps += 1
        time.sleep(1/self.fps)
        if self.verbose:
            print("STEP")
            print(f"action: {action}")
            print(f"reward: {reward}")
            print(f"terminated: {terminated}")
            print(f"truncated: {truncated}")
        if not self.termination_on_fall:
            terminated = self.steps == self.steps_to_reset
        return (obs[0], reward, terminated, truncated, obs[1])
    
    def reset(self, seed=None):
        obs, info = super().reset()
        self.steps = 0
        return obs, info



def build_env():
    env = TransformObservation(env=HumanoidBulletEnv(render=True))
    return env

