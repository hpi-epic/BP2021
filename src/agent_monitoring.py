import os
import time

import matplotlib.pyplot as plt
import numpy as np

import agent
import sim_market as sim


class Monitor():

	def __init__(self) -> None:
		self.enable_live_draws = True
		self.episodes = 500
		self.plot_interval = int(self.episodes / 10)
		# should get deprecated when introducing possibility to use multiple RL-agents
		self.path_to_modelfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + 'test_marketplace.dat'
		self.situation = 'linear'
		self.marketplace = sim.CircularEconomy() if self.situation == 'circular' else sim.ClassicScenario()
		self.agents = [agent.QLearningAgent(self.marketplace.observation_space.shape[0], self.marketplace.action_space.n, load_path=self.path_to_modelfile)]
		self.agent_colors = ['#0000ff']
		self.subfolder_path = 'plots_' + time.strftime('%Y%m%d-%H%M%S')
		self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + self.subfolder_path

	# helper functions
	def round_up(self, number, decimals=0) -> np.float64:
		multiplier = 10 ** decimals
		return np.ceil(number * multiplier) / multiplier

	def get_cmap(self, n, name='hsv') -> plt.cm.colors.LinearSegmentedColormap:
		return plt.cm.get_cmap(name, n + 1)

	def get_folder(self) -> str:
		# create folder with current timestamp to save diagrams at
		if not os.path.exists(self.folder_path):
			os.mkdir(self.folder_path)
			os.mkdir(self.folder_path + os.sep + 'histograms')
		return self.folder_path

	# configure the situation to be monitored
	def setup_monitoring(self, draw_enabled=None, new_episodes=None, new_plot_interval=None, new_modelfile=None, new_situation=None, new_marketplace=None, new_agents=None, new_subfolder_path=None) -> None:
		# doesn't look nice, but afaik only way to keep parameter list short
		if(draw_enabled is not None):
			self.enable_live_draws = draw_enabled
		if(new_episodes is not None):
			self.episodes = new_episodes
		if(new_plot_interval is not None):
			self.plot_interval = new_plot_interval
		if(new_modelfile is not None):
			self.path_to_modelfile = new_modelfile
		if(new_situation is not None):
			self.situation = new_situation
		if(new_marketplace is not None):
			self.marketplace = new_marketplace
		if(new_subfolder_path is not None):
			self.subfolder_path = new_subfolder_path
			self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + self.subfolder_path
		if(new_agents is not None):
			self.agents = new_agents
			color_map = self.get_cmap(len(self.agents))
			self.agent_colors = []
			for i in range(0, len(self.agents)):
				self.agent_colors.append(color_map(i))

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
	def create_histogram(self, rewards, filename='default') -> None:
		plt.xlabel('Reward', fontsize='18')
		plt.ylabel('Episodes', fontsize='18')
		plt.title('Cumulative Reward per Episode')
		# find the number of bins needed, we only use steps of 1000, assuming our agents are good bois :)
		plot_range = (0, self.round_up(int(self.metrics_maximum(rewards)), -3))
		plot_bins = int(int(plot_range[1]) / 1000)

		plt.hist(rewards, bins=plot_bins, color=self.agent_colors, rwidth=0.9, range=plot_range)
		plt.legend([a.name for a in self.agents])

		if self.enable_live_draws:  # pragma: no cover
			plt.draw()
			plt.pause(0.001)
		plt.savefig(fname=self.get_folder() + os.sep + 'histograms' + os.sep + 'episode_' + str(filename) + '.svg')

	def create_mean_line_plot(self, rewards, filename='default') -> None:
		# clear old plot completely
		plt.clf()
		# calculate mean rewards per self.plot_interval episodes for each agent
		episodes = np.arange(self.plot_interval, self.episodes + 1, self.plot_interval)
		mean_rewards = []
		for i in range(0, len(rewards)):
			mean_rewards.append([])
			for j in range(0, int(len(rewards[i]) / self.plot_interval)):
				mean_rewards[i].append(self.metrics_median(rewards[i][0:self.plot_interval * j + self.plot_interval]))

		# plot the mean rewards for each agent
		for means in mean_rewards:
			plt.plot(episodes, means, marker='o')

		plt.xlabel('Episodes', fontsize='18')
		plt.xticks(episodes)
		plt.ylabel('Mean Reward', fontsize='18')
		plt.title(f'Total Mean Reward calculated each {self.plot_interval} episodes')
		plt.legend([a.name for a in self.agents])
		plt.grid(True)
		if self.enable_live_draws:  # pragma: no cover
			plt.draw()
			plt.pause(0.001)
		plt.savefig(fname=self.get_folder() + os.sep + 'mean_rewards.svg')

	def run_marketplace(self) -> list:
		# initialize the rewards list with a list for each agent
		rewards = []
		for i in range(len(self.agents)):
			rewards.append([])

		for episode in range(1, self.episodes + 1):
			# reset the state once to be used by all agents
			default_state = self.marketplace.reset()

			for i in range(0, len(self.agents)):
				# reset marketplace, bit hacky, if you find a better solution feel free
				self.marketplace.reset()
				self.marketplace.state = default_state

				# reset values for all agents
				state = default_state
				episode_reward = 0
				is_done = False

				# run marketplace for this agent
				while not is_done:
					action = self.agents[i].policy(state)
					state, reward, is_done, _ = self.marketplace.step(action)
					episode_reward += reward

				# add the reward to the current agent's reward-Array
				rewards[i] += [episode_reward]

			# after all agents have run the episode
			if (episode % 100) == 0:
				print(f'Running {episode}th episode...')

			if (episode % self.plot_interval) == 0:
				self.create_histogram(rewards, episode)

		return rewards


monitor = Monitor()


def main() -> None:
	import agent
	monitor.setup_monitoring(draw_enabled=False, new_agents=[monitor.agents[0], agent.FixedPriceLEAgent(6, name='fixed_6'), agent.FixedPriceLEAgent(3, 'fixed_3')])
	print(f'Running', monitor.episodes, 'episodes')
	print(f'Plot interval is: {monitor.plot_interval}')
	print(f'Using modelfile: {monitor.path_to_modelfile}')
	print(f'The situation is: {monitor.situation}')
	print(f'The marketplace is: {monitor.marketplace}')
	print('Monitoring these agents:')
	for current_agent in monitor.agents:
		print(current_agent.name)

	rewards = monitor.run_marketplace()

	for i in range(len(rewards)):
		print(f'Statistics for agent: {monitor.agents[i].name}')
		print(f'The average reward over {monitor.episodes} episodes is: {str(monitor.metrics_average(rewards[i]))}')
		print(f'The median reward over {monitor.episodes} episodes is: {str(monitor.metrics_median(rewards[i]))}')
		print(f'The maximum reward over {monitor.episodes} episodes is: {str(monitor.metrics_maximum(rewards[i]))}')
		print(f'The minimum reward over {monitor.episodes} episodes is: {str(monitor.metrics_minimum(rewards[i]))}')

	print('Creating line plot for mean rewards...')
	monitor.create_mean_line_plot(rewards, 'mean_rewards')
	print(f'All plots were saved to {monitor.get_folder()}')


if __name__ == '__main__':  # pragma: no cover
	main()
