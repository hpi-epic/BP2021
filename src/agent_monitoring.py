from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import agent
import sim_market as sim
import utils as ut

# api tbd
situation = 'linear'
marketplace = sim.CircularEconomy() if situation == 'circular' else sim.ClassicScenario()
agent = agent.QLearningAgent(marketplace.observation_space.shape[0], marketplace.action_space.n, load_path='/Users/lion/Documents/HPI5/EPIC/BP2021/trainedModels/args.env-best_2760.77_marketplace.dat.dat')

# set up marketplace
number_episodes = int(input('Enter the number of episodes: '))

# run marketplace
def reset_episode(reward_per_episode, is_done, marketplace) -> Tuple[np.float64, bool, np.array]:
	reward_per_episode = 0
	is_done = False
	state = marketplace.reset()

	return reward_per_episode, is_done, state

def run_marketplace(number_episodes, marketplace, agent) -> list:

	# initialize marketplace
	reward_per_episode = 0
	rewards = []
	is_done = False

	for _ in range(number_episodes):

		reward_per_episode, is_done, state = reset_episode(reward_per_episode, is_done, marketplace)

		while not is_done:
			action = agent.policy(state)
			print(
				'This is the state:',
				marketplace.state,
				' and I will do ',
				action
			)
			state, reward, is_done, dict = marketplace.step(action)
			print('The agents profit this round is', reward)
			reward_per_episode += reward
		print(
			'The agents total profit of this episode is',
			reward_per_episode
		)
		rewards.append(reward_per_episode)

	return rewards

# metrics
def metrics_average(rewards, number_episodes) -> np.float64:
	return np.float64(sum(rewards) / number_episodes)

def metrics_median(rewards) -> np.float64:
	return np.median(np.array(rewards))

def metrics_maximum(rewards) -> np.float64:
	return np.max(np.array(rewards))

def metrics_minimum(rewards) -> np.float64:
	return np.min(np.array(rewards))



# visialize metrics


def main():
	rewards = run_marketplace(number_episodes, marketplace, agent)

	print(f'The average reward over {number_episodes} episodes is: '
		+ str(metrics_average(rewards, number_episodes)))
	print(f'The median reward over {number_episodes} episodes is: '
		+ str(metrics_median(rewards)))
	print(f'The maximum reward over {number_episodes} episodes is: '
		+ str(metrics_maximum(rewards)))
	print(f'The minimum reward over {number_episodes} episodes is: '
		+ str(metrics_minimum(rewards)))

if __name__ == '__main__':
	main()
