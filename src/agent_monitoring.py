import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import agent
import sim_market as sim


class Monitor():

	def __init__(self):
		self.episodes = 500
		self.histogram_plot_interval = int(self.episodes / 10)
		self.path_to_modelfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'testmodel' + os.sep + 'test_marketplace.dat'
		self.situation = 'linear'
		self.marketplace = sim.CircularEconomy() if self.situation == 'circular' else sim.ClassicScenario()
		self.agent = agent.QLearningAgent(self.marketplace.observation_space.shape[0], self.marketplace.action_space.n, load_path=self.path_to_modelfile)

	# helper functions
	def round_up(self, number, decimals=0):
		multiplier = 10 ** decimals
		return np.ceil(number * multiplier) / multiplier

	# configure the situation to be monitored
	def setup_monitoring(self, new_episodes=None, new_interval=None, new_modelfile=None, new_situation=None, new_marketplace=None, new_agent=None):
		# doesn't look nice, but afaik only way to keep parameter list short
		if(new_episodes is not None):
			self.episodes = new_episodes
		if(new_interval is not None):
			self.histogram_plot_interval = new_interval
		if(new_modelfile is not None):
			self.path_to_modelfile = new_modelfile
		if(new_situation is not None):
			self.situation = new_situation
		if(new_marketplace is not None):
			self.marketplace = new_marketplace
		if(new_agent is not None):
			self.agent = new_agent

	# metrics
	def metrics_average(self, rewards) -> np.float64:
		return np.mean(np.array(rewards))

	def metrics_median(self, rewards) -> np.float64:
		return np.median(np.array(rewards))

	def metrics_maximum(self, rewards) -> np.float64:
		return np.max(np.array(rewards))

	def metrics_minimum(self, rewards) -> np.float64:
		return np.min(np.array(rewards))

	# visualize metrics
	def create_histogram(self, rewards) -> None:
		plt.xlabel('Reward', fontsize='18')
		plt.ylabel('Episodes', fontsize='18')
		plt.hist(rewards, bins=10, color='#6e9bd1', align='mid', edgecolor='black', range=(0, self.round_up(int(self.metrics_maximum(rewards)), -3)))
		plt.draw()
		plt.pause(0.001)

	# run simulation
	def reset_episode(self, reward_per_episode, is_done, marketplace) -> Tuple[np.float64, bool, np.array]:
		reward_per_episode = 0
		is_done = False
		state = marketplace.reset()

		return reward_per_episode, is_done, state

	def run_marketplace(self, number_episodes, marketplace, agent) -> list:
		# initialize marketplace
		reward_per_episode = 0
		rewards = []
		is_done = False

		for episode in range(1, number_episodes + 1):

			reward_per_episode, is_done, state = self.reset_episode(reward_per_episode, is_done, marketplace)

			while not is_done:
				action = self.agent.policy(state)
				state, reward, is_done, _ = self.marketplace.step(action)
				reward_per_episode += reward

			rewards.append(reward_per_episode)

			if (episode % 100) == 0:
				print(f'Running {episode}th episode...')

			if (episode % self.histogram_plot_interval) == 0:
				self.create_histogram(rewards)

		return rewards


def main():
	monitor = Monitor()
	monitor.setup_monitoring(new_situation='linear')
	print(f'Running', monitor.episodes, 'episodes')
	print(f'Plot interval is:', monitor.histogram_plot_interval)
	print(f'Using modelfile: ' + monitor.path_to_modelfile)
	print(f'The situation is: ' + monitor.situation)
	print(f'The marketplace is:', monitor.marketplace)
	print(f'The agent is:', monitor.agent)
	rewards = monitor.run_marketplace(monitor.episodes, monitor.marketplace, monitor.agent)

	print(f'The average reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_average(rewards)))
	print(f'The median reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_median(rewards)))
	print(f'The maximum reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_maximum(rewards)))
	print(f'The minimum reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_minimum(rewards)))

	# show last histogram
	plt.draw()
	plt.pause(5)


if __name__ == '__main__':
	main()
