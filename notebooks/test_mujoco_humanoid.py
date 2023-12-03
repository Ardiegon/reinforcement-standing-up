import gym
from gym.utils.env_checker import check_env

env = gym.make("HumanoidStandup-v4", render_mode="human")
check_env(env)

env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()