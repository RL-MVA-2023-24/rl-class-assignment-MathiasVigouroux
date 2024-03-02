from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
#class ProjectAgent:
#    def act(self, observation, use_random=False):
#        return 0

#    def save(self, path):
#        pass

#    def load(self):
#        pass

class ActorCritic(nn.Module):
    def __init__(self, state_space=6, action_space=4, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_space, hidden_size)
        
        # Actor head
        self.action_layer = nn.Linear(hidden_size, action_space)
        
        # Critic head
        self.value_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        state = torch.relu(self.affine(state))
        
        action_probs = torch.softmax(self.action_layer(state), dim=-1)
        state_values = self.value_layer(state)
        
        return action_probs, state_values

class ProjectAgent:
    def __init__(self, state_space=6, action_space=4, lr=3e-4, gamma=0.99):
        self.model = ActorCritic(state_space, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.path = os.getcwd()+"/src/model.pt"

        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs, _ = self.model(state)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()
    
    def update(self, rewards, log_probs, state_values, next_state_value):
        # Calculate discounted rewards
        returns = []
        R = next_state_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        state_values = torch.stack(state_values)
        
        advantage = returns - state_values
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        self.optimizer.step()
    
    def save(self):
        self.path = os.getcwd()+"/src/model.pt"
        torch.save(self.model.state_dict(), self.path)
    
    def load(self):
        #self.path = os.getcwd()+"/model.pt"
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()


