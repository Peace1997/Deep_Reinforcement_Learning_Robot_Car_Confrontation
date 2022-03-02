from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import torch
import numpy as np

from maddpg.agent import DDPGAgent
from maddpg.maddpg import MADDPG
from utils import MultiAgentReplayBuffer
from simulator import ICRABattleField
import os
from torch.utils.tensorboard import SummaryWriter

def run(maddpg,max_episode, max_steps, batch_size,random_steps):
    
    episode_rewards = []
    total_episode = max_episode * max_steps
    step = 0
    states = env.reset()
    writer = SummaryWriter('./path/to/log')
    for episode in range(total_episode):
        
        episode_reward = 0
        
        #actions = maddpg.get_actions(states)
        if episode > random_steps:
            actions = maddpg.get_actions(states)
        else:
            actions = [torch.from_numpy(env.action_space[0].sample()),torch.from_numpy(env.action_space[0].sample())]

        step +=1
        
        next_states, rewards, dones, _ = maddpg.env.step(actions)
        episode_reward += np.mean(rewards)

        

        if dones or step == max_steps - 1:
            dones = [1 for _ in range(maddpg.num_agents)]
            maddpg.replay_buffer.push(states, actions, rewards, next_states, dones)
            episode_rewards.append(episode_reward)
            print("episode: {}  |  reward: {}  \n".format(episode, episode_reward))
            writer.add_scalar('Return', episode_reward, episode)
            states = maddpg.env.reset()
            step = 0
        else:
            dones = [0 for _ in range(maddpg.num_agents)]
            maddpg.replay_buffer.push(states, actions, rewards, next_states, dones)
            states = next_states 

            if len(maddpg.replay_buffer) > batch_size:
                maddpg.update(batch_size,writer,episode)                       
        #env.render()
        if episode % 2000000 == 0:
            os.makedirs('./model2_0_1/'+str(int(episode/2000000)))
            maddpg.agents[0].save_ID_R1('./model2_0_1/'+str(int(episode/2000000))+'/')
            maddpg.agents[1].save_ID_R2('./model2_0_1/'+str(int(episode/2000000))+'/')
env = ICRABattleField()
maddpg = MADDPG(env, 1000000)
run(maddpg,7000,3000,32,2000000)
