import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])



class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.pi.initialize_weights()
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q1.initialize_weights()
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2.initialize_weights()
    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
    def save(self,filename):
        torch.save(self.pi.state_dict(),filename+ 'actor.pkl')
        torch.save(self.q1.state_dict(), filename+ 'critic1.pkl')
        torch.save(self.q2.state_dict(), filename+ 'critic2.pkl')
    def red_load(self):
        self.pi.load_state_dict(torch.load('model1_4/5/actor.pkl'))
        self.q1.load_state_dict(torch.load('model1_4/5/critic1.pkl'))
        self.q2.load_state_dict(torch.load('model1_4/5/critic2.pkl'))
    def blue_load(self):
        self.pi.load_state_dict(torch.load('model1_4/5/actor.pkl'))
        self.q1.load_state_dict(torch.load('model1_4/5/critic1.pkl'))
        self.q2.load_state_dict(torch.load('model1_4/5/critic2.pkl'))

