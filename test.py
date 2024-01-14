import argparse

from src.agents.agent import get_model
from src.envs import get_env


def test(agent, env, args):
    obs, info = env.reset()

    for _ in range(1000):
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default="mujoco",
                        help='Choose from mujoco or pybullet')
    parser.add_argument('--total-timesteps', type=int, default=1000000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--load-model', default="default_model", help='Input model name (default: default_model)')
    parser.add_argument('--save-interval', default=100000, help="After what number of steps saves model")
    parser.add_argument('--agent-hparams', default=None, help="Yaml file with Agent parameters")
    return parser.parse_args()


def main(args):
    env = get_env(args.env_type)
    agent = get_model(get_env("mujoco"), args)

    test(agent, env, args=args)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
