'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.AC import ActorCriticAgent
from agent.Agent import HandAgent
from simulator import ICRABattleField
from utils import Action, ID_R1, ID_B1
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    description="Train the model in the ICRA 2019 Battlefield")
parser.add_argument("--seed", type=int, default=233, help="Random seed")
parser.add_argument("--enemy", type=str, default="hand",
                    help="The opposite agent type [AC, hand]")
parser.add_argument("--load_model", action='store_true',default=False,
                    help="Whether to load the trained model")
parser.add_argument("--load_actor_model_path", type=str,
                    default="actor.model", help="The path of trained actor model")
parser.add_argument("--load_critic_model_path", type=str,
                    default="critic.model", help="The path of trained actor model")
parser.add_argument("--save_actor_model_path", type=str,
                    default="actor.model", help="The path of trained actor model")
parser.add_argument("--save_critic_model_path", type=str,
                    default="critic.model", help="The path of trained critic model")
parser.add_argument("--epoch", type=int, default=50000,
                    help="Number of epoches to train")
parser.add_argument("--update_target_step", type=int, default=10,
                    help="After how many step, update the target?")
parser.add_argument("--update_model_step", type=int, default=10,
                    help="After how many step, update the model?")
args = parser.parse_args()


torch.random.manual_seed(args.seed)
torch.cuda.random.manual_seed(args.seed)
np.random.seed(args.seed)
#random.seed(args.seed)

agent = ActorCriticAgent()

if args.load_model:
    agent.load_model(args.load_actor_model_path,args.load_critic_model_path)
if args.enemy == "hand":
    agent2 = HandAgent()
elif args.enemy == "AC":
    agent2 = ActorCriticAgent()
    agent2.load_model(args.load_model_path)

env = ICRABattleField()
env.seed(args.seed)
critic_losses = []
actor_losses = []
rewards = []
for i_episode in range(1, args.epoch + 1):
    print("Epoch: [{}/{}]".format(i_episode, args.epoch))
    # Initialize the environment and state
    action = Action()
    pos = env.reset()
    if args.enemy == "hand":
        agent2.reset(pos)
    state, reward, done, info = env.step(action)
    for t in (range(2000)):
        # Other agent
        if args.enemy == "hand":
            env.set_robot_action(ID_B1, agent2.select_action(state[ID_B1]))
        elif args.enemy == "AC":
            env.set_robot_action(ID_B1, agent2.select_action(
                state[ID_B1], mode="max_probability"))

        # Select and perform an action
        state_map = agent.preprocess(state[ID_R1])
        a_m, a_t = agent.run_AC(state_map)
        if i_episode < 20000:
            action = agent.decode_action(a_m, a_t, state[ID_R1], "sample")
        else:
            action = agent.decode_action(a_m, a_t, state[ID_R1], "max_probability")
        # Step
        next_state, reward, done, info = env.step(action)
        tensor_next_state = agent.preprocess(next_state[ID_R1])

        # Store the transition in memory
        agent.push(state_map, tensor_next_state, [a_m, a_t], [reward])
        state = next_state
        state_map = tensor_next_state

        #env.render()
        # Perform one step of the optimization (on the target network)
        if done:
            break

    print("Simulation end in: {}:{:02d}, reward: {}".format(
        t//(60*30), t % (60*30)//30, env.reward))
    agent.memory.finish_epoch()
    critic_loss,actor_loss = agent.optimize_online()
    critic_losses.append(critic_loss)
    actor_losses.append(actor_loss)
    rewards.append(env.reward)

    writer = SummaryWriter('./path/to/log')
    writer.add_scalar('reward', env.reward, i_episode)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % args.update_model_step == 0:
        agent.save_model(args.save_actor_model_path,args.save_critic_model_path)
    if i_episode % args.update_target_step == 0:
        agent.actor_critic_soft_update()

print('Complete')
env.close()

plt.title("Critic Loss")
plt.xlabel("Epoch")
plt.ylabel("Critic Loss")
plt.plot(critic_losses)
plt.savefig("critic_loss.pdf")

plt.title("Actor Loss")
plt.xlabel("Epoch")
plt.ylabel("Actor Loss")
plt.plot(actor_losses)
plt.savefig("actor_loss.pdf")

plt.title("Reward")
plt.xlabel("Epoch")
plt.ylabel("Final reward")
plt.plot(rewards)
plt.savefig("reward.pdf")
