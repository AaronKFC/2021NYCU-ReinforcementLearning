# Spring 2021, IOC 5269 Reinforcement Learning
# HW2: REINFORCE with baseline and A2C

import gym
from itertools import count
import itertools
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

import matplotlib.pyplot as plt

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """

    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 64
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.fc0 = nn.Linear(self.state_dim, self.hidden_size)
        ### Actor_Net ###
        self.a_fc0 = nn.Linear(self.state_dim, self.hidden_size)
        self.a_fc1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.a_fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.a_fc3 = nn.Linear(self.hidden_size * 2, self.action_dim) 
        
        ### Baseline_Net ### (for estimating value function)
        self.c_fc0 = nn.Linear(self.state_dim, self.hidden_size)
        self.c_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_fc3 = nn.Linear(self.hidden_size, 1)
        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        ########## YOUR CODE HERE (3~5 lines) ##########
        # s = F.relu(self.fc0(state))
        ### Actor_branch ###
        x = F.relu(self.a_fc0(state))
        x = F.relu(self.a_fc1(x))
        x = F.relu(self.a_fc2(x))
        x = F.sigmoid(self.a_fc3(x))
        action_prob = x
        
        ### Baseline_branch ### (for estimating value function)
        y = F.relu(self.c_fc0(state))
        y = F.relu(self.c_fc1(y))
        y = F.relu(self.c_fc2(y))
        y = self.c_fc3(y)
        baseline_value = y
        ########## END OF YOUR CODE ##########
        return action_prob, baseline_value

    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.from_numpy(state).float().to(device)
        action_prob, self.state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()
        self.log_act_prob = m.log_prob(action)
        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), self.state_value))

        return action.item()

    def calculate_loss(self, reward, next_state, gamma=0.95):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        # R = 0
        # saved_actions = self.saved_actions
        # policy_loss = []
        # value_losses = []
        # returns = [] 

        ########## YOUR CODE HERE (8-15 lines) ##########
        next_value = 0
        
        value = self.state_value
        log_act_prob = self.log_act_prob
        
        # execute this function with "next_state" to get the "next_value" 
        self.select_action(next_state) 
        next_value = self.state_value.detach()
        
        baseline_loss = nn.MSELoss()
        value_loss = baseline_loss(value, reward+(gamma*next_value) )
        
        advantage = (reward + gamma*next_value)  - value
        policy_loss = -log_act_prob * advantage.detach()
        
        # print('policy_losses=',policy_loss)
        # print('value_losses=',value_loss)
        loss = policy_loss + value_loss
        
        ########## END OF YOUR CODE ##########

        return loss, policy_loss, value_loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode,
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''

    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # Record
    reward_lst = []

    try:
        # run inifinitely many episodes
        for i_episode in count(1):
            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0
            t = 0
            # Uncomment the following line to use learning rate scheduler
            scheduler.step()
        
            # For each episode, only run 9999 steps so that we don't 
            # infinite loop while learning

            ########## YOUR CODE HERE (10-15 lines) ##########
            for t in itertools.count(start=1):                
                action = model.select_action(state)
                next_state, reward, done, _ = env.step(action)

                ep_reward += reward
                state = next_state
                model.rewards.append(reward)
                
                loss, policy_loss, value_loss = model.calculate_loss(reward, next_state)
                # optimizer.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 3)
                # optimizer.step()
                
                ### Update Actor_Net ###
                optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                
                ### Update Behavior_Net ###
                optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                
                if done:
                    break
            
            model.clear_memory()
            ########## END OF YOUR CODE ##########

            # update EWMA reward and log the results
            ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
            reward_lst.append(ewma_reward)
            print('Ep {}\tLength:{:4d}\tR: {:.4f}\t ewma reward: {:.4f}'.format(
                i_episode, t, ep_reward, ewma_reward))

            # check if we have "solved" the cart pole problem
            if ewma_reward >= 172.:
                torch.save(model.state_dict(), './LunarLander_0.01.pth')
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(ewma_reward, t))
                break

    finally:

        plt.xlabel('# of episode')
        plt.ylabel('ewma_reward')
        plt.plot(reward_lst)
        plt.savefig("A2C_reward.png")


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''
    model = Policy()
    model.load_state_dict(torch.load('./{}'.format(name)))
    render = True
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20
    lr = 0.01
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test('LunarLander_0.01.pth')

