import gym
from gym.utils.env_checker import check_env

def build_env():
    env = gym.make("HumanoidStandup-v4", render_mode="human")
    check_env(env)
    env.action_space.seed(42)
    env.reset(seed=42)
    return env