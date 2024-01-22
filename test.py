import argparse
import matplotlib.pyplot as plt


from src.agents.agent import get_model
from src.envs import get_env


def test(agent, env, args):
    obs, info = env.reset()
    all_rewards = []

    for _ in range(1000):
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        all_rewards.append(reward)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

    if args.plot_rewards:
        plt.hist(all_rewards, bins=10, color='gray', edgecolor='black')
        plt.title('Nagrody zdobywane w 1000 kroków - Mujoco')
        plt.ylabel('Kroki')
        plt.xlabel('Przedział nagród')
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default="mujoco",
                        help='Choose from mujoco or pybullet')
    parser.add_argument('--total-timesteps', type=int, default=1000000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--load-model', default="default_model", help='Input model name (default: default_model)')
    parser.add_argument('--save-interval', default=100000, help="After what number of steps saves model")
    parser.add_argument('--agent-hparams', default=None, help="Yaml file with Agent parameters")
    parser.add_argument('--plot-rewards', '-p', action="store_true", help="plot histogram of rewards gathered in 1000 steps")
    return parser.parse_args()


def main(args):
    env = get_env(args.env_type)
    agent = get_model(get_env("mujoco"), args) # agent always get mujoco because of synchronization with length 376 observation vector 

    test(agent, env, args=args)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
