from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
import random
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def act(self, observation, use_random=False): #only the actor will do an action
        device = "cuda" if next(self.actor.parameters()).is_cuda else "cpu"
        observation_tensor = torch.Tensor(observation).unsqueeze(0).to(device)

        with torch.no_grad():
            action_probabilities = self.actor(observation_tensor)
            if use_random:
                # Select a random action with uniform probability
                action = torch.randint(0, len(action_probabilities.squeeze()), (1,)).item()
            else:
                # Sample an action from the probability distribution output by the actor
                distribution = torch.distributions.Categorical(action_probabilities)
                action = distribution.sample().item()

        return action

    def save(self, filepath):
      checkpoint = {
          'actor_state_dict': self.actor.state_dict(),
          'critic_state_dict': self.critic.state_dict()

      }
      path = filepath + '/AC_model.pt'
      torch.save(checkpoint, path)
      print(f"Model saved to {path}")
      return



    def load(self):
      device = torch.device('cpu')
      filepath = os.getcwd()
      path = filepath + '/AC_model.pt'
      checkpoint = torch.load(path)
      self.actor, self.critic = self.create_actor_critic_networks(
            env.observation_space.shape[0],
            env.action_space.n,
            256,  
            device
        )
      # Load the state dicts
      self.actor.load_state_dict(checkpoint['actor_state_dict'])
      self.critic.load_state_dict(checkpoint['critic_state_dict'])
      print(f"Model loaded from {path}")
      return


    ## MODEL ARCHITECTURE

    def create_actor_critic_networks_initial(self, state_dim, n_action, nb_neurons, device): #action_critic_network
      # Actor Network
      actor = torch.nn.Sequential(
          nn.Linear(state_dim, nb_neurons),
          nn.ReLU(),
          nn.Linear(nb_neurons, nb_neurons),
          nn.ReLU(),
          nn.Linear(nb_neurons, n_action),
          nn.Softmax(dim=-1)  # Use Softmax for action probability distribution
      ).to(device)
      # Critic Network
      critic = torch.nn.Sequential(
          nn.Linear(state_dim, nb_neurons),
          nn.ReLU(),
          nn.Linear(nb_neurons, nb_neurons),
          nn.ReLU(),
          nn.Linear(nb_neurons, 1)  # Outputs a single value estimating the state value
      ).to(device)
      return actor, critic



    def create_actor_critic_networks(self, state_dim, n_action, nb_neurons, device):
      '''
      I add layer normalization (used in altegrad, and a dropout layer seen in Theoretical foudnation of deep learning)
      I try to introduce some regularization
      '''
      # Actor Network
      actor = torch.nn.Sequential(
          nn.Linear(state_dim, nb_neurons),
          nn.ReLU(),
          nn.Linear(nb_neurons, nb_neurons),
          nn.ReLU(),
          nn.Linear(nb_neurons, nb_neurons),
          nn.ReLU(),
          nn.Linear(nb_neurons, nb_neurons),
          nn.ReLU(),
          nn.Dropout(p=0.2),  # Dropout for regularization
          nn.Linear(nb_neurons, n_action),
          nn.Softmax(dim=-1)
      ).to(device)
      # Critic Network
      critic = torch.nn.Sequential(
          nn.Linear(state_dim, nb_neurons),
          nn.LeakyReLU(),  # Different activation function
          nn.Linear(nb_neurons, nb_neurons),
          nn.LeakyReLU(),
          nn.Linear(nb_neurons, nb_neurons),
          nn.LeakyReLU(),
          nn.Linear(nb_neurons, nb_neurons),
          nn.LeakyReLU(),
          nn.LayerNorm(nb_neurons),  # Batch normalization
          nn.Linear(nb_neurons, 1)
      ).to(device)
      return actor, critic





    def train(self):
    ## CONFIGURE NETWORK
    # Actor-Critic config
      config = {
          'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.98,
          'buffer_size': 100000,
          'batch_size': 64,  # Typically smaller batch size for online updates
          'update_freq': 1,  # Update frequency for the actor and critic networks
          'critic_learning_rate': 0.001,  # Learning rate for critic
          'actor_learning_rate': 0.0001,  # Learning rate for actor
          'gradient_steps' : 100,
      }

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      print('Using device:', device)

      # Initialize actor and critic networks
      #print(env.observation_space.shape[0], env.action_space.n)
      self.actor, self.critic = self.create_actor_critic_networks(
          env.observation_space.shape[0],
          env.action_space.n,
          256,  # Example: number of neurons in hidden layers
          device
      )



      # Optimizers for actor and critic
      actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'])
      critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_learning_rate'])

      

      # Loss function for critic updates
      critic_loss_fn = torch.nn.MSELoss()

      ## TRAINING LOOP
      max_episodes = 200
      episode = 0
      val_episode = 50 #the episode when we start to save by looking at the validation loss
      previous_val = 0.
      for episode in range(max_episodes):
          state, _ = env.reset()
          episode_cum_reward = 0

          for _ in range(config['gradient_steps']):
              # Convert state to tensor
              state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

              # Actor decides on action
              action_prob = self.actor(state_tensor)
              distribution = torch.distributions.Categorical(action_prob)
              action = distribution.sample()

              # Take action
              #print(env.step(action.item()))
              next_state, reward, done, trunc, _ = env.step(action.item())
              next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

              # Critic evaluates decision
              value = self.critic(state_tensor)
              next_value = self.critic(next_state_tensor).detach()

              # Compute advantage and critic loss
              td_target = reward + config['gamma'] * next_value * (1 - int(done))
              advantage = td_target - value
              critic_loss = critic_loss_fn(value, td_target)

              # Update critic
              critic_optimizer.zero_grad()
              critic_loss.backward()
              critic_optimizer.step()

              # Update actor using policy gradient
              actor_loss = -(distribution.log_prob(action) * advantage.detach())
              #print(actor_loss)
              actor_optimizer.zero_grad()
              actor_loss.backward()
              actor_optimizer.step()

              episode_cum_reward += reward
              state = next_state


          if episode > val_episode:
            validation_score = evaluate_HIV(agent=self, nb_episode=1)
          else:
            validation_score = 0.


          print(f"Episode {episode + 1}/{max_episodes}, Total Reward: {episode_cum_reward}, Validation score: {validation_score}")

          if validation_score > previous_val:

            print('Improvement, saving a model')
            previous_val = validation_score
            path = os.getcwd()
            self.save(path)


      print("Training complete.")



