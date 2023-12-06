from src.envs import get_env

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC

env = get_env("mujoco")
# check_env(env, warn=True, skip_render_check=True)
model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_pendulum")


obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()