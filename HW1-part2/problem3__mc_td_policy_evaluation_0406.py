# Spring 2021, IOC 5269 Reinforcement Learning
# HW1-PartII: First-Visit Monte-Carlo and Temporal-difference policy evaluation

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import sys
import itertools


env = gym.make("Blackjack-v0")

def sample_episode(policy, env):
    states, actions, rewards = [], [], []
    # Initialize the gym environment
    observation = env.reset()
    while True:
        states.append(observation)
        
        # select an action using apply_policy()
        action = apply_policy(observation)
        actions.append(action)
        
        # perform the action in the environment according to apply_policy(), move to the next state, and receive reward
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        
        # Break if the state is terminal
        if done:
             break
    return states, actions, rewards


def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for a given policy using first-visit Monte-Carlo sampling
        
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
        
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
    
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate sample returns
            3. Iterate and update the value function
        ----------
        
    """
    # value function
    V = defaultdict(float)
    
    ##### FINISH TODOS HERE #####
    N = defaultdict(int)

    # for _ in range(num_episodes):
    for i in range(num_episodes+1):
        # print out training progress
        if i % 1000 == 0:
            print(f"\rEpisode {i}/{num_episodes}  ", end="")
            sys.stdout.flush()
            
        # generate the epsiode and store the states and rewards
        states, _, rewards = sample_episode(policy, env)
        returns = 0
        
        # for each step, store rewards to a variable R and states to S, and calculate returns as a sum of rewards
        for t in reversed(range(len(states))):
            R = rewards[t]
            S = states[t]
            returns += R
            
            # Now to perform first visit MC, check if the episode is visited for the first time, 
            # if yes, take the average of returns and assign the value of the state as an average of returns
            if S not in states[:t]:
                N[S] += 1
                V[S] += (returns - V[S]) / N[S]
    #############################
    return V


def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for the given policy using TD(0)
    
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
    
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate TD errors
            3. Iterate and update the value function
        ----------
    """
    # value function
    V = defaultdict(float)

    ##### FINISH TODOS HERE #####
    for i in range(1,num_episodes+1):
        # print out training progress
        if i % 1000 == 0:
            print(f"\rEpisode {i}/{num_episodes}  ", end="")
            sys.stdout.flush()
        
        # Reset the environment
        S = env.reset()
        A = policy(S)
        
        # One step in the environment
        for t in itertools.count():
            # iterating steps
            next_S, R, done, _ = env.step(A)
            
            # obtain the next_action based on next_state
            next_A = policy(next_S)
            
            # conduct TD learning
            td_target = R + gamma * V[next_S]
            td_error = td_target - V[S]
            alpha = 0.5
            V[S] += alpha * td_error 
    
            if done:
                break
                
            A = next_A
            S = next_S   
    #############################
    return V

    

def plot_value_function(V, title="Value Function"):
    """
        Plots the value function as a surface plot.
        (Credit: Denny Britz)
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    
    
def apply_policy(observation):
    """
        A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_mc_10k, title="10,000 Steps")
    print('V_mc_10k: done')
    V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_mc_500k, title="500,000 Steps")
    print('V_mc_500k: done')

    V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_td0_10k, title="10,000 Steps")
    print('V_td0_10k: done')
    V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_td0_500k, title="500,000 Steps")
    print('V_td0_500k: done')



