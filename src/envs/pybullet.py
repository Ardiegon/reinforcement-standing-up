from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv 
from gym.utils.env_checker import check_env

def build_env():
    env = HumanoidBulletEnv(render=True)
    check_env(env)
    env.seed(42)
    env.action_space.seed(42)
    env.reset()
    return env

