import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

import agent
import sim_market
import utils as ut


class Monitor():
	"""
	A Monitor() monitors given agents on a marketplace, recording metrics such as median and maximum rewards.

	When the run is finished, diagrams will be created in the 'monitoring' folder. \\
	The Monitor() can be customized using setup_monitoring().
	"""

	def __init__(self) -> None:
		# Do not change the values in here! They are assumed in tests. Instead use setup_monitoring()!
		assert os.path.exists(self.get_modelfile_path('CircularEconomy_QLearningCEAgent.dat')), 'the default modelfile \'CircularEconomy_QLearningCEAgent.dat\' does not exist'
		self.enable_live_draw = True
		self.episodes = 500
		self.plot_interval = 50
		self.marketplace = sim_market.CircularEconomy()
		self.agents = [agent.QLearningCEAgent(self.marketplace.observation_space.shape[0], self.get_action_space(), load_path=self.get_modelfile_path('CircularEconomy_QLearningCEAgent.dat'))]
		self.agent_colors = ['#0000ff']
		self.subfolder_name = 'plots_' + time.strftime('%Y%m%d-%H%M%S')
		self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + self.subfolder_name

	# helper functions
	def round_up(self, number, decimals=0) -> np.float64:
		multiplier = 10 ** decimals
		return np.ceil(number * multiplier) / multiplier

	def round_down(self, number, decimals=0) -> np.float64:
		multiplier = 10 ** decimals
		return np.floor(number * multiplier) / multiplier

	def get_cmap(self, n, name='hsv') -> plt.cm.colors.LinearSegmentedColormap:
		"""
		Return a colormap containing a distinct color for each monitored agent to be used in the diagrams.

		Args:
			n (int): How many colors should be generated.
			name (str, optional): The type of colormap that should be generated. Defaults to 'hsv'.

		Returns:
			plt.cm.colors.LinearSegmentedColormap: The filled colormap.
		"""
		return plt.cm.get_cmap(name, n + 1)

	def get_folder(self) -> str:
		"""
		Return the folder where all diagrams of the current run are saved.

		Returns:
			str: The folder name
		"""
		# create folder with current timestamp to save diagrams at
		if not os.path.exists(self.folder_path):
			os.mkdir(self.folder_path)
			os.mkdir(self.folder_path + os.sep + 'histograms')
		return self.folder_path

	def get_modelfile_path(self, model_name) -> str:
		"""
		Get the full path to a modelfile in the \'monitoring\' folder.

		Args:
			model_name (str): The name of the .dat modelfile.

		Returns:
			str: The full path to the modelfile.
		"""
		assert str.endswith(model_name, '.dat'), 'the modelfile must be a .dat file'
		full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + model_name
		assert os.path.exists(full_path), 'the specified modelfile does not exist'
		return full_path

	def get_action_space(self) -> int:
		"""
		Return the size of the action space in the self.marketplace.

		Returns:
			int: The size of the action space
		"""
		n_actions = 1
		if isinstance(self.marketplace.action_space, gym.spaces.Discrete):
			n_actions = self.marketplace.action_space.n
		else:
			for id in range(0, len(self.marketplace.action_space)):
				n_actions *= self.marketplace.action_space[id].n
		return n_actions

	def update_agents(self, agents) -> None:
		"""
		Update the self.agents to the new agents provided.

		Args:
			agents (list of tuples of agent classes and lists): What agents to monitor. Must be tuples where the first entry is the class of the agent and the second entry is a list of arguments for its initialization. Arguments are read left to right, arguments cannot be skipped. Each agent will generate data points in the diagrams. Defaults to None.

		Raises:
			RuntimeError: Raised if the modelfile provided does not match the Market/Agent-type provided.
		"""
		# All agents must be of the same type
		assert all(isinstance(agent_tuple, tuple) for agent_tuple in agents), 'agents must be a list of tuples'
		assert all(len(agent_tuple) == 2 for agent_tuple in agents), 'the list entries in agents must have size 2 ([agent_class, arguments])'
		assert all(issubclass(agent_tuple[0], agent.Agent) for agent_tuple in agents), 'the first entry in each agent-tuple must be an agent class in agent.py'
		assert all(isinstance(agent_tuple[1], list) for agent_tuple in agents), 'the second entry in each agent-tuple must be a list of arguments'
		assert all(issubclass(agent_tuple[0], agent.CircularAgent) == issubclass(agents[0][0], agent.CircularAgent) for agent_tuple in agents), 'the agents must all be of the same type (Linear/Circular)'
		assert issubclass(agents[0][0], agent.CircularAgent) == isinstance(self.marketplace, sim_market.CircularEconomy), 'the agent and marketplace must be of the same economy type (Linear/Circular)'

		self.agents = []

		# Instantiate all agents. If they are not rule-based, use the marketplace parameters accordingly
		for current_agent in agents:
			if issubclass(current_agent[0], agent.RuleBasedAgent):
				self.agents.append(agent.Agent.custom_init(agent.Agent, current_agent[0], current_agent[1]))
			elif not issubclass(current_agent[0], agent.RuleBasedAgent):
				try:
					assert len(current_agent[1]) == 1 or len(current_agent[1]) == 2 and isinstance(current_agent[1][1], str), 'the first argument for an reinforcement lerner needs to be a modelfile, the second one is an optional name (str)'
					assert isinstance(current_agent[1][0], str), 'the modelfile must be of type str'

					agent_name = 'q_learning' if len(current_agent[1]) == 1 else current_agent[1][1]
					self.agents.append(current_agent[0](self.marketplace.observation_space.shape[0], self.get_action_space(), load_path=self.get_modelfile_path(current_agent[1][0]), name=agent_name))
				except RuntimeError:  # pragma: no cover
					raise RuntimeError('the modelfile is not compatible with the agent you tried to instantiate')
			else:  # pragma: no cover
				assert False, current_agent[0] + 'is neither a rule_based nor a reinforcement_learning agent'

		# set a color for each agent
		color_map = self.get_cmap(len(self.agents))
		self.agent_colors = []
		for agent_id in range(0, len(self.agents)):
			self.agent_colors.append(color_map(agent_id))

	def setup_monitoring(self, enable_live_draw=None, episodes=None, plot_interval=None, marketplace=None, agents=None, subfolder_name=None) -> None:
		"""
		Configure the current monitoring session.

		Args:
			enable_live_draw (bool, optional): Whether or not diagrams should be displayed on screen when drawn. Defaults to None.
			episodes (int, optional): The number of episodes to run. Defaults to None.
			plot_interval (int, optional): After how many episodes a new data point/plot should be generated. Defaults to None.
			marketplace (sim_market class, optional): What marketplace to run the monitoring on. Defaults to None.
			agents (list of tuples of agent classes and lists): What agents to monitor.
			Must be tuples where the first entry is the class of the agent and the second entry is a list of arguments for its initialization.
			Arguments are read left to right, arguments cannot be skipped.
			The first argument must exist and be the path to the modelfile for the agent, the second is optional and the name the agent should have.
			Each agent will generate data points in the diagrams. Defaults to None.
			subfolder_name (str, optional): The name of the folder to save the diagrams in. Defaults to None.
		"""
		# doesn't look nice, but afaik only way to keep parameter list short
		if(enable_live_draw is not None):
			assert isinstance(enable_live_draw, bool), 'enable_live_draw must be a Boolean'
			self.enable_live_draw = enable_live_draw
		if(episodes is not None):
			assert isinstance(episodes, int), 'episodes must be of type int'
			self.episodes = episodes
		if(plot_interval is not None):
			assert isinstance(plot_interval, int), 'plot_interval must be of type int'
			self.plot_interval = plot_interval

		if(marketplace is not None):
			assert issubclass(marketplace, sim_market.SimMarket), 'the marketplace must be a subclass of sim_market'
			self.marketplace = marketplace()
			# The agents have not been changed, we reuse the old agents
			if(agents is None):
				print('Warning: Your agents are being overwritten by new instances of themselves!')
				agents = [(type(current_agent), [f'{type(monitor.marketplace).__name__}_{type(current_agent).__name__}.dat']) for current_agent in self.agents]
			self.update_agents(agents)

		# marketplace has not changed but agents have
		elif(agents is not None):
			self.update_agents(agents)

		if(subfolder_name is not None):
			assert isinstance(subfolder_name, str), 'subfolder_name must be of type string'
			self.subfolder_name = subfolder_name
			self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + self.subfolder_name

	def get_configuration(self) -> dict:
		"""
		Return the configuration of the current monitor.

		Returns:
			dict: A dict containing the configuration (=class variables)
		"""
		configuration = {}
		configuration['enable_live_draw'] = self.enable_live_draw
		configuration['episodes'] = self.episodes
		configuration['plot_interval'] = self.plot_interval
		configuration['marketplace'] = self.marketplace
		configuration['agents'] = self.agents
		configuration['agent_colors'] = self.agent_colors
		configuration['subfolder_name'] = self.subfolder_name
		configuration['folder_path'] = self.folder_path
		return configuration

	def get_episode_rewards(self, all_step_rewards) -> list:
		"""
		Accumulates all rewards per episode

		Args:
			all_step_rewards (list of list of floats): Contains a list per agent containing float rewards for the episode

		Returns:
			list of list of floats: List of accumulated rewards per episode per agent
		"""
		episode_rewards = []
		for agent_reward in all_step_rewards:
			episode_rewards.append([])
			curr_sum = 0
			for reward_index in range(len(agent_reward)):
				curr_sum += agent_reward[reward_index]
				if(reward_index % ut.EPISODE_LENGTH == ut.EPISODE_LENGTH - 1):
					# one episode is over
					episode_rewards[-1] += [curr_sum]
					curr_sum = 0
		return episode_rewards

	# metrics
	def metrics_average(self, rewards) -> np.float64:
		return np.mean(np.array(rewards))

	def metrics_median(self, rewards) -> np.float64:
		return np.median(np.array(rewards))

	def metrics_maximum(self, rewards) -> np.float64:
		return np.max(np.array(rewards))

	def metrics_minimum(self, rewards) -> np.float64:
		return np.min(np.array(rewards))

	# def metrics_average_in_episode(self, rewards) -> np.float64:
	# 	return sum(rewards) / (len(rewards) * ut.EPISODE_LENGTH)

	# visualize metrics
	def create_histogram(self, rewards, filename='default') -> None:
		"""
		Create a histogram sorting rewards into bins of 1000.

		Args:
			rewards (array of arrays of int): An array containing an array of ints for each monitored agent.
			filename (str, optional): The name of the output file, format will be .svg. Defaults to 'default'.
		"""
		plt.clf()
		plt.xlabel('Reward', fontsize='18')
		plt.ylabel('Episodes', fontsize='18')
		plt.title('Cumulative Reward per Episode')
		# find the number of bins needed, we only use steps of 1000, assuming our agents are good bois :)
		plot_range = self.round_down(int(self.metrics_minimum(rewards)), -3), self.round_up(int(self.metrics_maximum(rewards)), -3)
		plot_bins = int(int(np.abs(plot_range[0]) + plot_range[1]) / 1000)

		plt.hist(rewards, bins=plot_bins, color=self.agent_colors, rwidth=0.9, range=plot_range)
		plt.legend([a.name for a in self.agents])

		if self.enable_live_draw:  # pragma: no cover
			plt.draw()
			plt.pause(0.001)
		plt.savefig(fname=self.get_folder() + os.sep + 'histograms' + os.sep + filename + '.svg')

	def create_statistics_plots(self, rewards) -> None:
		"""
		For each of our metrics, calculate the running value each self.plot_interval and plot it as a line graph.

		Current metrics: Average, Maximum, Median, Minimum.

		Args:
			rewards ([list of list of float]): An array containing an array of ints for each monitored agent.
		"""
		metrics_functions = [self.metrics_average, self.metrics_maximum, self.metrics_median, self.metrics_minimum]  # , self.metrics_average_in_episode
		metrics_names = ['Average', 'Maximum', 'Median', 'Minimum']  # , 'Average in episode'
		x_axis_episodes = np.arange(self.plot_interval, self.episodes + 1, self.plot_interval)

		for function in range(len(metrics_functions)):
			# calculate <metric> rewards per self.plot_interval episodes for each agent
			metric_rewards = []
			for agent_rewards_id in range(0, len(rewards)):
				metric_rewards.append([])
				for starting_index in range(0, int(len(rewards[agent_rewards_id]) / self.plot_interval)):
					metric_rewards[agent_rewards_id].append(metrics_functions[function](rewards[agent_rewards_id][0:self.plot_interval * starting_index + self.plot_interval]))
			self.create_line_plot(x_axis_episodes, metric_rewards, metrics_names[function])

	def create_line_plot(self, x_values, y_values, metric_name='no name provided') -> None:
		"""Create a line plot with the given rewards data.

		Args:
			x_values (list of ints): Defines x-values of datapoints. Must have same length as y_values.
			y_values (list of list of ints): Defines y-values of datapoints, one array per monitored agent. Must have same length as episodes.
			metric_name (str, optional): Used for naming the y-axis, diagram and output file. Defaults to 'no name provided'.
		"""
		assert len(x_values) == int(self.episodes / self.plot_interval), 'x_values must have self.episodes / self.plot_interval many items'
		assert len(y_values) == len(self.agents), 'y_values must have one entry per agent'
		assert all(len(agent_y_value) == int(self.episodes / self.plot_interval) for agent_y_value in y_values), 'y_values must have self.episodes / self.plot_interval many items'
		print(f'Creating line plot for {metric_name} rewards...')
		# clear old plot completely
		plt.clf()
		filename = metric_name + '_rewards.svg'
		# plot the metric rewards for each agent
		for index in range(len(y_values)):
			plt.plot(x_values, y_values[index], marker='o', color=self.agent_colors[index])

		plt.xlabel('Episodes', fontsize='18')
		# array containing the values to be plotted on the x axis, equally spaced each self.plot_interval
		plt.xticks(np.arange(0, self.episodes + 1, self.plot_interval))
		plt.ylabel(f'{metric_name} Reward', fontsize='18')
		plt.title(f'Overall {metric_name} Reward calculated each {self.plot_interval} episodes')
		plt.legend([a.name for a in self.agents])
		plt.grid(True)
		if self.enable_live_draw:  # pragma: no cover
			plt.draw()
			plt.pause(0.001)
		plt.savefig(fname=self.get_folder() + os.sep + filename)

	def run_marketplace(self) -> list:
		"""
		Run the marketplace with the given monitoring configuration.

		Automatically produces histograms, but not metric diagrams.

		Returns:
			list: A list with a list of rewards for each agent
		"""

		# initialize the rewards list with a list for each agent
		rewards = []
		for i in range(len(self.agents)):
			rewards.append([])

		# all_steps_rewards = []
		# for i in range(len(self.agents)):
		# 	all_steps_rewards.append([])

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
					state, step_reward, is_done, _ = self.marketplace.step(action)
					episode_reward += step_reward
					# this gives us a higher flexibility in terms of what metrics we would like to use
					# all_steps_rewards[i] += [step_reward]

				# removing this will decrease our performance when we still want to do live drawing
				# could think about a caching strategy for live drawing
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
	# monitor.setup_monitoring(enable_live_draw=False, agents=[(agent.QLearningCEAgent, ['CircularEconomy_QLearningCEAgent.dat']), (agent.FixedPriceCEAgent, [(4,6)])])
	print('Live Drawing enabled:', monitor.enable_live_draw)
	print('Episodes:', monitor.episodes)
	print(f'Plot interval: {monitor.plot_interval}')
	print(f'Marketplace: {type(monitor.marketplace).__name__}')
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

	monitor.create_statistics_plots(rewards)
	print(f'All plots were saved to {monitor.get_folder()}')


if __name__ == '__main__':  # pragma: no cover
	main()
