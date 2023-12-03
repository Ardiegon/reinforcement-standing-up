import gym
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv 

env = HumanoidBulletEnv(render=True)

# Inicjalizacja środowiska
env.action_space.seed(42)

env.reset()

# Przykładowa pętla interakcji ze środowiskiem
for _ in range(1000):  # Możesz dostosować ilość kroków
    observation, reward, done, _ = env.step(env.action_space.sample())

    input()
    if done:
        print("woop")
        env.reset()

env.close()