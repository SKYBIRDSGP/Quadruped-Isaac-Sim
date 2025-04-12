import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Proximal Policy Optimization on our Quadruped

class Actor(nn.Module): 
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.linear(state_dim, 800),
            nn.ReLU,
            nn.linear(800, 500),
            nn.ReLU,
            nn.linear(500, 300),
            nn.ReLU,
            nn.linear(300, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.linear(state_dim, 800),
            nn.ReLU,
            nn.linear(800, 500),
            nn.ReLU,
            nn.linear(500, 300),
            nn.ReLU,
            nn.linear(300, 1)
            )
    
    def forward(self, state):
        return self.net(state)
    
class PPO(object):
    def __init__(self, state_dim, action_dim, max_action):
        ## for the Actor Model ##
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        ## for the Critic Model ##
        self.critic = Critic(state_dim, action_dim, max_action).to(device)
        self.critic_target = Critic(state_dim, action_dim, max_action).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
        self.max_action = max_action

        def select_action(self, state):
            pass

        def train(self, replay_buffer, iterations, discount=0.98, clip=0.75):
            pass