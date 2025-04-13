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
            nn.Linear(state_dim, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 800),
            nn.ReLU(),
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
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
        self.critic = Critic(state_dim).to(device)
        self.critic_target = Critic(state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        
        self.max_action = max_action

        def select_action(self, state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state).cpu().data.numpy().flatten()

        def train(self, iterations, replay_buff, batch_size, learning_rate, clip_range, discount = 0.98, gae_param = 0.98, T = 2048):
            
            for it in range(iterations):

                x, y, u, r, d = replay_buff.sample(batch_size)

                state = torch.FloatTensor(x).to(device)
                action = torch.FloatTensor(u).to(device)
                nxt_state = torch.FloatTensor(y).to(device)
                done = torch.FloatTensor(1-d).to(device)
                reward = torch.FloatTensor(r).to(device)

                # Getting the current values from the Critic
                value = self.critic(state)
                nxt_value = self.critic(nxt_state)

                # Calculating the delta
                delta = reward + discount*nxt_value*done - value

                # Compute advantage using GAE
                advantage = torch.zeros_like(reward).to(device)
                gae = 0

                for t in reversed(range(len(reward))):
                    gae = delta[t] + discount*gae_param*done[t]*gae
                    advantage[t] = gae
                returns = advantage + value

                # Log probablity of the actions for current policy
                mu = self.actor(state)
                dist = torch.distributions.Normal(mu, 1.0)
                log_prob = dist.log_prob(action).sum(axis=-1)

                # Log probablity of the actions for old policy
                with torch.no_grad():
                    mu_old = self.actor_target(state)
                    dist_old = torch.distributions.Normal(mu_old, 1.0)
                    log_prob_old = dist_old.log_prob(action).sum(axis=-1)
                
                ratio = torch.exp(log_prob - log_prob_old)

                # Clipped Loss
                clip_adv = torch.clamp(ratio, 1-clip_range, 1+clip_range)*advantage
                actor_loss = -torch.min(ratio*advantage, clip_adv).mean()

                # Critic Loss
                critic_loss = F.mse_loss(value, returns.detach())

                # Optimizer step
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Update the old policy
                self.actor_target.load_state_dict(self.actor.state_dict())
            
        def save(self, filename, directory):
            torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
            torch.save(self.critic.state_dict(), '%s/%s_actor.pth' % (directory, filename))

        def load(self, filename, directory):
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%critic.pth' % (directory, filename)))            