import os
import argparse
import datetime
import itertools

from torch.utils.tensorboard import SummaryWriter

from src.hparams import HParamsSac
from src.agents import SAC, ReplayMemory
from src.envs import get_env


def train(env, agent, memory, writer, args):
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state) 

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _, _= env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)
            state = next_state

            if total_numsteps%args.save_interval==0:
                agent.save_checkpoint(args.env_type, str(total_numsteps))


        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default="mujoco",
                    help='Choose from mujoco or pybullet')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=2000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
    parser.add_argument('--save-interval', default=50000, help="After what number of steps saves model")
    parser.add_argument('--agent-hparams', default=None, help="Yaml file with Agent parameters")
    return parser.parse_args()

def main(args):
    env = get_env(args.env_type)
    hparams=HParamsSac(args.agent_hparams)
    print(hparams)
    agent = SAC(env.observation_space.shape[0], env.action_space, hparams)
    memory = ReplayMemory()
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_type,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    train(env=env, agent=agent, writer=writer, memory=memory, args = args)
    
    env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
