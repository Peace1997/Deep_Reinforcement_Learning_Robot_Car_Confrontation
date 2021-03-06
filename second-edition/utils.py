import numpy as np 
import random
from collections import deque
import torch

from collections import namedtuple

ID_R1 = 0
ID_R2 = 1
ID_B1 = 2


COLOR_BLACK = (0.0, 0.0, 0.0)
COLOR_RED = (0.8, 0.0, 0.0)
COLOR_BLUE = (0.1, 0.5, 0.8)
COLOR_WHITE = (1.0, 1.0, 1.0)
COLOR_LIGHT_RED = (0.9, 0.4, 0.4, 1.0)
COLOR_LIGHT_BLUE = (0.4, 0.4, 9.0, 1.0)

GROUP_RED = "red"
GROUP_BLUE = "blue"

UserData = namedtuple("UserData", ["type", "id"])


class MultiAgentReplayBuffer:
    
    def __init__(self, num_agents, max_size):
        self.max_size = max_size
        self.num_agents = num_agents
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array(reward), next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        obs_batch = [[] for _ in range(self.num_agents)]  # [ [states of agent 1], ... ,[states of agent n] ]    ]
        indiv_action_batch = [[] for _ in range(self.num_agents)] # [ [actions of agent 1], ... , [actions of agent n]]
        indiv_reward_batch = [[] for _ in range(self.num_agents)]
        next_obs_batch = [[] for _ in range(self.num_agents)]

        global_state_batch = []
        global_next_state_batch = []
        global_actions_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)


        for experience in batch:
            state, action, reward, next_state, done = experience
            
            for i in range(self.num_agents):
                obs_i = state[i]
                action_i = action[i]
                reward_i = reward[i]
                next_obs_i = next_state[i]
            
                obs_batch[i].append(obs_i)
                indiv_action_batch[i].append(action_i)
                indiv_reward_batch[i].append(reward_i)
                next_obs_batch[i].append(next_obs_i)

            global_state_batch.append(np.concatenate(state))
            global_actions_batch.append(torch.cat(action))
            global_next_state_batch.append(np.concatenate(next_state))
            done_batch.append(done)
        
        return obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)