import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import agent
import sim_market as sim

# api tbd
situation = 'linear'
marketplace = sim.CircularEconomy() if situation == 'circular' else sim.ClassicScenario()
agent = agent.QLearningAgent(marketplace.observation_space.shape[0], marketplace.action_space.n, load_path='/Users/lion/Documents/HPI5/EPIC/BP2021/trainedModels/classic_scenario_end.dat.dat')

# constants
NUMBER_OF_EPISODES = 500
HISTOGRAM_PLOT_INTERVAL = int(NUMBER_OF_EPISODES / 10)

# helper functions
def round_up(number, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(number * multiplier) / multiplier

# metrics
def metrics_average(rewards) -> np.float64:
	return np.mean(np.array(rewards))

def metrics_median(rewards) -> np.float64:
	return np.median(np.array(rewards))

def metrics_maximum(rewards) -> np.float64:
	return np.max(np.array(rewards))

def metrics_minimum(rewards) -> np.float64:
	return np.min(np.array(rewards))

# visualize metrics
def create_histogram(rewards) -> None:
	plt.xlabel('Reward', fontsize='18')
	plt.ylabel('Episodes', fontsize='18')
	plt.hist(rewards, bins=10, color='#6e9bd1', align='mid' ,edgecolor='purple', range=(0, round_up(int(metrics_maximum(rewards)), -3)))
	plt.draw()
	plt.pause(0.001)

# run simulation
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

	for episode in range(1, number_episodes + 1):

		reward_per_episode, is_done, state = reset_episode(reward_per_episode, is_done, marketplace)

		while not is_done:
			action = agent.policy(state)
			state, reward, is_done, _ = marketplace.step(action)
			reward_per_episode += reward

		rewards.append(reward_per_episode)

		if (episode % 100) == 0:
			print(f'Running {episode}th episode...')

		if (episode % HISTOGRAM_PLOT_INTERVAL) == 0:
			create_histogram(rewards)

	return rewards


def main():
	# track time
	start_time = time.time()

	rewards = run_marketplace(NUMBER_OF_EPISODES, marketplace, agent)

	print(f'Finished!\nRunning marketplace for {NUMBER_OF_EPISODES} episodes took {time.time() - start_time} seconds\n')

	print(f'The average reward over {NUMBER_OF_EPISODES} episodes is: '
		+ str(metrics_average(rewards)))
	print(f'The median reward over {NUMBER_OF_EPISODES} episodes is: '
		+ str(metrics_median(rewards)))
	print(f'The maximum reward over {NUMBER_OF_EPISODES} episodes is: '
		+ str(metrics_maximum(rewards)))
	print(f'The minimum reward over {NUMBER_OF_EPISODES} episodes is: '
		+ str(metrics_minimum(rewards)))

	# show last histogram
	plt.draw()
	plt.pause(5)


if __name__ == '__main__':
	main()
