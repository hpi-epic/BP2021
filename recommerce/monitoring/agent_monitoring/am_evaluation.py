import os

import matplotlib.pyplot as plt
import numpy as np

import recommerce.monitoring.agent_monitoring.am_configuration as am_configuration


class Evaluator():
	"""
	The Evaluator is used together with the `agent_monitoring.Monitor()` and is able to create different graphs from rewards it gets from it.
	"""
	def __init__(self, configuration: am_configuration.Configurator):
		self.configurator = configuration

	def evaluate_session(self, analyses: dict, episode_numbers: list = None):
		"""
		Print statistics for monitored agents and create statistics-plots.

		Args:
			analyses (list): A list for every analyzed vendor. Contains a dict with the properties as keys and all samples of this property.
			episode_numbers (list of int): The training stages the empirical distributions belong to.
			If it is None, a prior functionality is used.
		"""
		# Print the statistics
		for index, analysis in enumerate(analyses):
			if episode_numbers is None:
				print(f'\nStatistics for agent: {self.configurator.agents[index].name}')
			else:
				print(f'\nStatistics for episode {episode_numbers[index]}')

			for property in analysis:
				samples = analysis[property]
				print('%40s: %7.2f (mean), %7.2f (median), %7.2f (std), %7.2f (min), %7.2f (max)' % (
					property, np.mean(samples), np.median(samples), np.std(samples), np.min(samples), np.max(samples)))

		print()
		# self._create_statistics_plots(rewards)
		if episode_numbers is not None:
			for property_name in analyses[0]:
				samples = [analysis[property_name] for analysis in analyses]
				self._create_violin_plot(samples, episode_numbers, f'Progress of {property_name}')
		print(f'All plots were saved to {os.path.abspath(self.configurator.folder_path)}')

	# visualize metrics
	def create_histogram(self, rewards: list, is_last_histogram: bool, filename: str = 'default_histogram.svg') -> None:
		"""
		Create a histogram sorting rewards into bins of 1000.

		Args:
			rewards (array of arrays of int): An array containing an array of ints for each monitored agent.
			is_last_histogram (bool): States that only the last histogram should be plotted.
			filename (str): The name of the output file, format will be .svg. Defaults to 'default'.
		"""
		if (self.configurator.enable_live_draw or is_last_histogram) is False:
			return
		assert all(len(curr_reward) == len(rewards[0]) for curr_reward in rewards), 'all rewards-arrays must be of the same size'

		plt.clf()
		plt.xlabel('Reward', fontsize='18')
		plt.ylabel('Episodes', fontsize='18')
		plt.title('Cumulative Reward per Episode')

		# find the number of bins needed, we only use steps of 1000, assuming our agents are good bois :)
		plot_lower_bound = np.floor(int(np.min(rewards)) * 1e-3) / 1e-3
		plot_upper_bound = np.ceil(int(np.max(rewards)) * 1e-3) / 1e-3
		plot_bins = int(np.abs(plot_lower_bound) + plot_upper_bound) // 1000
		x_ticks = np.arange(plot_lower_bound, plot_upper_bound + 1, 1000)

		plt.hist(rewards, bins=plot_bins, color=self.configurator.agent_colors, rwidth=0.9,
			range=(plot_lower_bound, plot_upper_bound), edgecolor='black')
		plt.xticks(x_ticks)
		plt.legend([a.name for a in self.configurator.agents])

		if self.configurator.enable_live_draw:
			plt.draw()
			plt.pause(0.001)
		if is_last_histogram:
			plt.savefig(fname=os.path.join(self.configurator.get_folder(), filename))

	def _create_statistics_plots(self, rewards: list) -> None:
		"""
		For each of our metrics, calculate the running value each self.plot_interval and plot it as a line graph.

		Current metrics: Average, Maximum, Median, Minimum.

		Args:
			rewards ([list of list of float]): An array containing an array of ints for each monitored agent.
		"""
		# the functions that should be called to calculate the given metric
		metric_functions = [np.mean, np.median, np.max, np.min]
		# the name both the file as well as the plot title will have
		metric_names = ['Average', 'Median', 'Maximum', 'Minimum']
		# what kind of metric it is: Overall means the values are calculated from 0-episode, Episode means from previous episode-episode
		metric_types = ['Overall', 'Overall', 'Episode', 'Episode']

		x_axis_episodes = np.arange(self.configurator.plot_interval, self.configurator.episodes + 1, self.configurator.plot_interval)

		for function in range(len(metric_functions)):
			# calculate <metric> rewards per self.plot_interval episodes for each agent
			metric_rewards = []
			for agent_rewards_id in range(len(rewards)):
				metric_rewards.append([])
				for start_index in range(int(len(rewards[agent_rewards_id]) / self.configurator.plot_interval)):
					if metric_types[function] == 'Overall':
						metric_rewards[agent_rewards_id].append(
							metric_functions[function](rewards[agent_rewards_id][:self.configurator.plot_interval * (start_index + 1)]))
					elif metric_types[function] == 'Episode':
						metric_rewards[agent_rewards_id].append(
							metric_functions[function](
								rewards[agent_rewards_id][self.configurator.plot_interval * (start_index):self.configurator.plot_interval * (start_index + 1)]))
					else:  # pragma: no cover
						raise RuntimeError(f'this metric_type is unknown: {metric_types[function]}')
			self._create_line_plot(x_axis_episodes, metric_rewards, metric_names[function], metric_types[function])

	def _create_line_plot(self, x_values: list, y_values: list, metric_name: str, metric_type: str) -> None:
		"""Create a line plot with the given rewards data.

		Args:
			x_values (list of ints): Defines x-values of datapoints. Must have same length as y_values.
			y_values (list of list of ints): Defines y-values of datapoints, one array per monitored agent. Must have same length as episodes.
			metric_name (str): Used for naming the y-axis, diagram and output file.
			metric_type (str): What kind of "message" should be displayed at the top of the diagram.
		"""
		assert len(x_values) == int(self.configurator.episodes / self.configurator.plot_interval), \
			'x_values must have self.episodes / self.plot_interval many items'
		assert len(y_values) == len(self.configurator.agents), 'y_values must have one entry per agent'
		assert all(len(agent_y_value) == int(self.configurator.episodes / self.configurator.plot_interval) for agent_y_value in y_values), \
			'y_values must have self.episodes / self.plot_interval many items'
		print(f'Creating line plot for {metric_name} rewards...')

		plt.clf()
		filename = f'{metric_name}_rewards.svg'
		# plot the metric rewards for each agent
		for index in range(len(y_values)):
			plt.plot(x_values, y_values[index], marker='o', color=self.configurator.agent_colors[index])

		plt.xlabel('Episodes', fontsize='18')
		plt.ylabel(f'{metric_name} Reward', fontsize='18')

		if metric_type == 'Episode':
			plt.title(f'{metric_name} Reward within each previous {self.configurator.plot_interval} episodes')
		elif metric_type == 'Overall':
			plt.title(f'Overall {metric_name} Reward calculated each {self.configurator.plot_interval} episodes')
		else:
			raise RuntimeError(f'this metric_type is unknown: {metric_type}')

		plt.legend([a.name for a in self.configurator.agents])
		plt.grid(True)
		plt.savefig(fname=os.path.join(self.configurator.get_folder(), filename))

	def _create_violin_plot(self, all_rewards, episode_numbers, title='Add a title for this violin plot'):
		"""
		This method generates a violinplot to visualize the training progress of the agent.
		Provide the empirical distributions and it will not just show the mean, min and max,
		but also the distribution at the provided episode numbers.

		Args:
			all_rewards (list of lists): each entry contains samples for the empirical probability distribution
			episode_numbers (list of int): the training stages the empirical distributions belong to
		"""
		assert isinstance(all_rewards, list), f'all_rewards must be of type list, but is {type(all_rewards)}'
		assert isinstance(episode_numbers, list), f'episode_numbers must be of type list, but is {type(all_rewards)}'
		assert len(all_rewards) > 0, f'all_rewards should not be an empty list: {all_rewards}'
		assert len(all_rewards) == len(episode_numbers), 'the len of the rewards and the episode numbers must be the same'
		assert all(isinstance(rewards_on_training_stage, list) for rewards_on_training_stage in all_rewards), \
			'all_rewards must contain lists only'
		assert all(isinstance(episode_number, int) for episode_number in episode_numbers), \
			'episode_numbers must contain ints only'

		plt.clf()
		plt.violinplot(all_rewards, episode_numbers, showmeans=True, widths=450)
		plt.title(title)
		plt.xlabel('Learned Episodes')
		plt.ylabel('Reward Density')
		savepath = os.path.join(self.configurator.get_folder(), title.replace(' ', '_').replace('/', '_') + '.svg')
		plt.savefig(fname=savepath)


if __name__ == '__main__':  # pragma: no cover
	raise RuntimeError('agent_monitoring can only be run from `am_monitoring.py`')
