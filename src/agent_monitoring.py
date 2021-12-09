import os
import time

import matplotlib.pyplot as plt
import numpy as np

import agent
import sim_market as sim


class Monitor():
	"""
	A Monitor() monitors given agents on a marketplace, recording metrics such as median and maximum rewards.

	When the run is finished, diagrams will be created in the 'monitoring' folder. \\
	The Monitor() can be customized using setup_monitoring().
	"""

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
		"""Return a colormap containing a distinct color for each monitored agent to be used in the diagrams."""
		return plt.cm.get_cmap(name, n + 1)

	def get_folder(self) -> str:
		"""Return the folder where all diagrams of the current run are saved."""
		# create folder with current timestamp to save diagrams at
		if not os.path.exists(self.folder_path):
			os.mkdir(self.folder_path)
			os.mkdir(self.folder_path + os.sep + 'histograms')
		return self.folder_path

	def setup_monitoring(self, draw_enabled=None, episodes=None, plot_interval=None, modelfile=None, situation=None, marketplace=None, agents=None, subfolder_path=None) -> None:
		"""
		Configure the current monitoring session.

		### Parameters:
		- ``draw_enabled`` (bool, optional): Whether or not diagrams should be diplayed on screen.
		- ``episodes`` (int, optional): The number of episodes to run.
		- ``plot_interval`` (int, optional): After how many episodes a new data point/plot should be generated.
		- ``modelfile`` (str, optional): Path to the file containing the model for a RL-agent.
		- ``situation`` (str, optional): 'linear' or 'circular', which market situation should be played.
		- ``marketplace`` (sim_market instance, optional): What marketplace to run the monitoring on.
		- ``agents`` (array of agent instances, optional): What agents to monitor. Each agent will generate data points in the diagrams
		- ``subfolder_path`` (str, optional): The name of the folder to save the diagrams in. Defaults to 'plots_currentTime'
		"""
		# doesn't look nice, but afaik only way to keep parameter list short
		if(draw_enabled is not None):
			self.enable_live_draws = draw_enabled
		if(episodes is not None):
			self.episodes = episodes
		if(plot_interval is not None):
			self.plot_interval = plot_interval
		if(modelfile is not None):
			self.path_to_modelfile = modelfile
		if(situation is not None):
			self.situation = situation
		if(marketplace is not None):
			self.marketplace = marketplace
		if(subfolder_path is not None):
			self.subfolder_path = subfolder_path
			self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + self.subfolder_path
		if(agents is not None):
			self.agents = agents
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
		"""
		Create a histogram sorting rewards into bins of 1000.

		### Parameters:
		- ``rewards`` (array of arrays of int): An array containing an array of ints for each monitored agent.
		- ``filename`` (str, optional): The name of the output file, format will be ``.svg``.
		"""
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
		plt.savefig(fname=self.get_folder() + os.sep + 'histograms' + os.sep + filename + '.svg')

	def create_stat_plots(self, rewards) -> None:
		"""
		For each of our metrics, calculate the running value each self.plot_interval and plot it as a line graph.

		Current metrics: Average, Maximum, Median, Minimum.
		### Parameters:
		- ``rewards`` (array of arrays of int): An array containing an array of ints for each monitored agent.
		"""
		metrics_functions = [self.metrics_average, self.metrics_maximum, self.metrics_median, self.metrics_minimum]
		metrics_names = ['Average', 'Maximum', 'Median', 'Minimum']
		x_axis_episodes = np.arange(self.plot_interval, self.episodes + 1, self.plot_interval)

		for function in range(len(metrics_functions)):
			# calculate <metric> rewards per self.plot_interval episodes for each agent
			metric_rewards = []
			for i in range(0, len(rewards)):
				metric_rewards.append([])
				for j in range(0, int(len(rewards[i]) / self.plot_interval)):
					metric_rewards[i].append(metrics_functions[function](rewards[i][0:self.plot_interval * j + self.plot_interval]))
			self.create_line_plot(x_axis_episodes, metric_rewards, metrics_names[function])

	def create_line_plot(self, episodes, metric_rewards, metric_name='default') -> None:
		"""
		Create a line plot with the given rewards data.

		### Parameters:
		``episodes`` (array of ints): Defines x-values of datapoints. Must have same length as ``metric_rewards``
		``metric_rewards`` (array of array of ints): Defines y-values of datapoints, one array per monitored agent. Must have same length as ``episodes``.
		``metric_name`` (str, optional): Used for naming the y-axis, diagram and output file.
		"""
		print(f'Creating line plot for {metric_name} rewards...')
		# clear old plot completely
		plt.clf()
		filename = metric_name + '_rewards.svg'
		# plot the metric rewards for each agent
		for values in metric_rewards:
			plt.plot(episodes, values, marker='o')

		plt.xlabel('Episodes', fontsize='18')
		plt.xticks(episodes)
		plt.ylabel(f'{metric_name} Reward', fontsize='18')
		plt.title(f'Overall {metric_name} Reward calculated each {self.plot_interval} episodes')
		plt.legend([a.name for a in self.agents])
		plt.grid(True)
		if self.enable_live_draws:  # pragma: no cover
			plt.draw()
			plt.pause(0.001)
		plt.savefig(fname=self.get_folder() + os.sep + filename)

	def run_marketplace(self) -> list:
		"""
		Run the marketplace with the given monitoring configuration.

		Automatically produces histograms, but not metric diagrams.
		"""
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
				self.create_histogram(rewards, 'episode_' + str(episode))

		return rewards


monitor = Monitor()


def main() -> None:
	import agent
	monitor.setup_monitoring(draw_enabled=False, agents=[monitor.agents[0], agent.FixedPriceLEAgent(6, name='fixed_6'), agent.FixedPriceLEAgent(3, 'fixed_3')])
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

	monitor.create_stat_plots(rewards)
	print(f'All plots were saved to {monitor.get_folder()}')


if __name__ == '__main__':  # pragma: no cover
	main()
