import os
import time

import matplotlib.pyplot as plt
import numpy as np

import agents.vendors as vendors
import market.sim_market as sim_market


class Monitor():
	"""
	A Monitor() monitors given agents on a marketplace, recording metrics such as median and maximum rewards.

	When the run is finished, diagrams will be created in the 'monitoring' folder. \\
	The Monitor() can be customized using setup_monitoring().
	"""

	def __init__(self) -> None:
		# Do not change the values in here when setting up a session! They are assumed in tests. Instead use setup_monitoring()!
		self.enable_live_draw = True
		self.episodes = 500
		self.plot_interval = 50
		self.marketplace = sim_market.CircularEconomyMonopolyScenario()
		default_agent = vendors.QLearningCEAgent
		default_modelfile = f'{type(self.marketplace).__name__}_{default_agent.__name__}'
		assert os.path.exists(self.get_modelfile_path(default_modelfile)), f'the default modelfile does not exist: {default_modelfile}'
		self.agents = [default_agent(self.marketplace.observation_space.shape[0], self.get_action_space(), load_path=self.get_modelfile_path(default_modelfile))]
		self.agent_colors = [(0.0, 0.0, 1.0, 1.0)]
		self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + 'plots_' + time.strftime('%Y%m%d-%H%M%S')

	def get_folder(self) -> str:
		"""
		Return the folder where all diagrams of the current run are saved.

		Returns:
			str: The folder name
		"""
		# create folder with current timestamp to save diagrams at
		if not os.path.exists(self.folder_path):
			os.mkdir(self.folder_path)
			os.mkdir(os.path.join(self.folder_path, 'histograms'))
		return self.folder_path

	def get_modelfile_path(self, model_name: str) -> str:
		"""
		Get the full path to a modelfile in the 'monitoring' folder.

		Args:
			model_name (str): The name of the .dat modelfile.

		Returns:
			str: The full path to the modelfile.
		"""
		model_name += '.dat'
		full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'monitoring', model_name))
		assert os.path.exists(full_path), f'the specified modelfile does not exist: {full_path}'
		return full_path

	def get_action_space(self) -> int:
		"""
		Return the size of the action space in the self.marketplace.

		Returns:
			int: The size of the action space
		"""
		n_actions = 1
		if isinstance(self.marketplace, sim_market.CircularEconomy):
			for id in range(len(self.marketplace.action_space)):
				n_actions *= self.marketplace.action_space[id].n
		else:
			n_actions = self.marketplace.action_space.n
		return n_actions

	def _update_agents(self, agents: list) -> None:
		"""
		Update the self.agents to the new agents provided.

		Args:
			agents (list of tuples of agent classes and lists): What agents to monitor. Must be tuples where the first entry is the class of the agent and the second entry is an optional list of arguments for its initialization. Each agent will generate data points in the diagrams. See `setup_monitoring()` for more info. Defaults to None.

		Raises:
			RuntimeError: Raised if the modelfile provided does not match the Market/Agent-type provided.
		"""
		# All agents must be of the same type
		assert all(isinstance(agent_tuple, tuple) for agent_tuple in agents), 'agents must be a list of tuples'
		assert all(len(agent_tuple) == 2 for agent_tuple in agents), 'the list entries in agents must have size 2 ([agent_class, arguments])'
		assert all(issubclass(agent_tuple[0], vendors.Agent) for agent_tuple in agents), 'the first entry in each agent-tuple must be an agent class in `vendors.py`'
		assert all(isinstance(agent_tuple[1], list) for agent_tuple in agents), 'the second entry in each agent-tuple must be a list'
		assert all(issubclass(agent_tuple[0], vendors.CircularAgent) == issubclass(agents[0][0], vendors.CircularAgent) for agent_tuple in agents), 'the agents must all be of the same type (Linear/Circular)'
		assert issubclass(agents[0][0], vendors.CircularAgent) == isinstance(self.marketplace, sim_market.CircularEconomy), 'the agent and marketplace must be of the same economy type (Linear/Circular)'

		self.agents = []

		# Instantiate all agents. If they are not rule-based, use the marketplace parameters accordingly
		for current_agent in agents:
			if issubclass(current_agent[0], vendors.RuleBasedAgent):
				self.agents.append(vendors.Agent.custom_init(vendors.Agent, current_agent[0], current_agent[1]))
			elif issubclass(current_agent[0], vendors.QLearningAgent):
				try:
					assert (0 <= len(current_agent[1]) <= 2), 'the argument list for a RL-agent must have length between 0 and 2'
					assert all(isinstance(argument, str) for argument in current_agent[1]), 'the arguments for a RL-agent must be of type str'

					agent_modelfile = f'{type(self.marketplace).__name__}_{current_agent[0].__name__}'
					agent_name = 'q_learning'
					# no arguments
					if len(current_agent[1]) == 0:
						pass
					# only name argument
					elif len(current_agent[1]) == 1 and not str.endswith(current_agent[1][0], '.dat'):
						# get implicit modelfile name
						agent_name = current_agent[1][0]
					# only modelfile argument
					elif len(current_agent[1]) == 1 and str.endswith(current_agent[1][0], '.dat'):
						agent_modelfile = current_agent[1][0][:-4]
					# both arguments
					elif len(current_agent[1]) == 2:
						assert str.endswith(current_agent[1][0], '.dat'), f'if two arguments are provided, the first one must be the modelfile. Arg1: {current_agent[1][0]}, Arg2: {current_agent[1][1]}'
						agent_modelfile = current_agent[1][0][:-4]
						agent_name = current_agent[1][1]
					# this should never happen due to the asserts before, but you never know
					else:  # pragma: no cover
						raise RuntimeError('invalid arguments provided')

					# create the agent
					self.agents.append(current_agent[0](self.marketplace.observation_space.shape[0], self.get_action_space(), load_path=self.get_modelfile_path(agent_modelfile), name=agent_name))
				except RuntimeError:  # pragma: no cover
					raise RuntimeError('the modelfile is not compatible with the agent you tried to instantiate')
			else:  # pragma: no cover
				assert False, f'{current_agent[0]} is neither a RuleBased nor a QLearning agent'

		# set a color for each agent
		color_map = plt.cm.get_cmap('hsv', len(self.agents) + 1)
		self.agent_colors = [color_map(agent_id) for agent_id in range(len(self.agents))]

	def setup_monitoring(self, enable_live_draw: bool = None, episodes: int = None, plot_interval: int = None, marketplace: sim_market.SimMarket = None, agents: list = None, subfolder_name: str = None) -> None:
		# sourcery skip: extract-duplicate-method
		"""
		Configure the current monitoring session.

		Args:
			enable_live_draw (bool, optional): Whether or not diagrams should be displayed on screen when drawn. Defaults to None.
			episodes (int, optional): The number of episodes to run. Defaults to None.
			plot_interval (int, optional): After how many episodes a new data point/plot should be generated. Defaults to None.
			marketplace (sim_market class, optional): What marketplace to run the monitoring on. Defaults to None.
			agents (list of tuples of agent classes and lists): What agents to monitor. Each entry must be a tuple of a valid agent class and a list of optional arguments, where a .dat modelfile and/or a name for the agent can be specified. Modelfile defaults to \'marketplaceClass_AgentClass.dat\', Name defaults to \'q_learning\'
			Must be tuples where the first entry is the class of the agent and the second entry is a list of arguments for its initialization.
			Arguments are read left to right, arguments cannot be skipped.
			The first argument must exist and be the path to the modelfile for the agent, the second is optional and the name the agent should have.
			Each agent will generate data points in the diagrams. Defaults to None.
			subfolder_name (str, optional): The name of the folder to save the diagrams in. Defaults to None.
		"""
		if(enable_live_draw is not None):
			assert isinstance(enable_live_draw, bool), 'enable_live_draw must be a Boolean'
			self.enable_live_draw = enable_live_draw
		if(episodes is not None):
			assert isinstance(episodes, int), 'episodes must be of type int'
			assert episodes > 0, 'episodes must not be 0'
			self.episodes = episodes
		if(plot_interval is not None):
			assert isinstance(plot_interval, int), 'plot_interval must be of type int'
			assert plot_interval > 0, 'plot_interval must not be 0'
			assert plot_interval <= self.episodes, f'plot_interval must be <= episodes, or no plots can be generated. Episodes: {self.episodes}. Plot_interval: {plot_interval}'
			self.plot_interval = plot_interval

		if(marketplace is not None):
			assert issubclass(marketplace, sim_market.SimMarket), 'the marketplace must be a subclass of SimMarket'
			self.marketplace = marketplace()
			# If the agents have not been changed, we reuse the old agents
			if(agents is None):
				print('Warning: Your agents are being overwritten by new instances of themselves!')
				agents = [(type(current_agent), [f'{type(self.marketplace).__name__}_{type(current_agent).__name__}']) for current_agent in self.agents]
			self._update_agents(agents)

		# marketplace has not changed but agents have
		elif(agents is not None):
			self._update_agents(agents)

		if(subfolder_name is not None):
			assert isinstance(subfolder_name, str), f'subfolder_name must be of type str: {type(subfolder_name)}, {subfolder_name}'
			self.folder_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'monitoring', subfolder_name)

	def get_configuration(self) -> dict:
		"""
		Return the configuration of the current monitor as a dictionary.

		Returns:
			dict: A dict containing the configuration (=class variables)
		"""
		return {
			'enable_live_draw': self.enable_live_draw,
			'episodes': self.episodes,
			'plot_interval': self.plot_interval,
			'marketplace': self.marketplace,
			'agents': self.agents,
			'agent_colors': self.agent_colors,
			'folder_path': self.folder_path,
		}

	# visualize metrics
	def create_histogram(self, rewards: list, filename: str) -> None:
		"""
		Create a histogram sorting rewards into bins of 1000.

		Args:
			rewards (array of arrays of int): An array containing an array of ints for each monitored agent.
			filename (str): The name of the output file, format will be .svg.
		"""
		assert all(len(curr_reward) == len(rewards[0]) for curr_reward in rewards), 'all rewards-arrays must be of the same size'

		filename += '.svg'
		plt.clf()
		plt.xlabel('Reward', fontsize='18')
		plt.ylabel('Episodes', fontsize='18')
		plt.title('Cumulative Reward per Episode')

		# find the number of bins needed, we only use steps of 1000, assuming our agents are good bois :)
		plot_lower_bound = np.floor(int(np.min(rewards)) * 1e-3) / 1e-3
		plot_upper_bound = np.ceil(int(np.max(rewards)) * 1e-3) / 1e-3
		plot_bins = int(np.abs(plot_lower_bound) + plot_upper_bound) // 1000
		x_ticks = np.arange(plot_lower_bound, plot_upper_bound + 1, 1000)

		plt.hist(rewards, bins=plot_bins, color=self.agent_colors, rwidth=0.9, range=(plot_lower_bound, plot_upper_bound))
		plt.xticks(x_ticks)
		plt.legend([a.name for a in self.agents])

		if self.enable_live_draw:  # pragma: no cover
			plt.draw()
			plt.pause(0.001)
		plt.savefig(fname=os.path.join(self.get_folder(), 'histograms', filename))

	def create_statistics_plots(self, rewards: list) -> None:
		"""
		For each of our metrics, calculate the running value each self.plot_interval and plot it as a line graph.

		Current metrics: Average, Maximum, Median, Minimum.

		Args:
			rewards ([list of list of float]): An array containing an array of ints for each monitored agent.
		"""
		# the functions that should be called to calculate the given metric
		metric_functions = [np.mean, np.max, np.median, np.min]
		# the name both the file as well as the plot title will have
		metric_names = ['Average', 'Median', 'Maximum', 'Minimum']  # , 'Average in episode'
		# what kind of metric it is: Overall means the values are calculated from 0-episode, Episode means from previousEpisode-Episode
		metric_types = ['Overall', 'Overall', 'Episode', 'Episode']

		x_axis_episodes = np.arange(self.plot_interval, self.episodes + 1, self.plot_interval)

		for function in range(len(metric_functions)):
			# calculate <metric> rewards per self.plot_interval episodes for each agent
			metric_rewards = []
			for agent_rewards_id in range(len(rewards)):
				metric_rewards.append([])
				for starting_index in range(int(len(rewards[agent_rewards_id]) / self.plot_interval)):
					if metric_types[function] == 'Overall':
						metric_rewards[agent_rewards_id].append(
							metric_functions[function](rewards[agent_rewards_id][:self.plot_interval * (starting_index + 1)]))
					elif metric_types[function] == 'Episode':
						metric_rewards[agent_rewards_id].append(
							metric_functions[function](rewards[agent_rewards_id][self.plot_interval * (starting_index):self.plot_interval * (starting_index + 1)]))
					else:  # pragma: no cover
						raise RuntimeError(f'this metric_type is unknown: {metric_types[function]}')
			self.create_line_plot(x_axis_episodes, metric_rewards, metric_names[function], metric_types[function])

	def create_line_plot(self, x_values: list, y_values: list, metric_name: str, metric_type: str) -> None:
		"""Create a line plot with the given rewards data.

		Args:
			x_values (list of ints): Defines x-values of datapoints. Must have same length as y_values.
			y_values (list of list of ints): Defines y-values of datapoints, one array per monitored agent. Must have same length as episodes.
			metric_name (str): Used for naming the y-axis, diagram and output file.
			metric_type (str): What kind of "message" should be displayed at the top of the diagram.
		"""
		assert len(x_values) == int(self.episodes / self.plot_interval), 'x_values must have self.episodes / self.plot_interval many items'
		assert len(y_values) == len(self.agents), 'y_values must have one entry per agent'
		assert all(len(agent_y_value) == int(self.episodes / self.plot_interval) for agent_y_value in y_values), 'y_values must have self.episodes / self.plot_interval many items'
		print(f'Creating line plot for {metric_name} rewards...')

		plt.clf()
		filename = metric_name + '_rewards.svg'
		# plot the metric rewards for each agent
		for index in range(len(y_values)):
			plt.plot(x_values, y_values[index], marker='o', color=self.agent_colors[index])

		plt.xlabel('Episodes', fontsize='18')
		plt.ylabel(f'{metric_name} Reward', fontsize='18')

		if metric_type == 'Overall':
			plt.title(f'Overall {metric_name} Reward calculated each {self.plot_interval} episodes')
		elif metric_type == 'Episode':
			plt.title(f'{metric_name} Reward within each previous {self.plot_interval} episodes')
		else:  # pragma: no cover
			raise RuntimeError(f'this metric_type is unknown: {metric_type}')

		plt.legend([a.name for a in self.agents])
		plt.grid(True)
		plt.savefig(fname=os.path.join(self.get_folder(), filename))

	def run_marketplace(self) -> list:
		"""
		Run the marketplace with the given monitoring configuration.

		Automatically produces histograms, but not metric diagrams.

		Returns:
			list: A list with a list of rewards for each agent
		"""

		# initialize the rewards list with a list for each agent
		rewards = [[] for _ in range(len(self.agents))]

		for episode in range(1, self.episodes + 1):
			# reset the state once to be used by all agents
			default_state = self.marketplace.reset()

			for i in range(len(self.agents)):
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


def run_monitoring_session(monitor: Monitor = Monitor()) -> None:
	"""
	Run a monitoring session with a configured Monitor() and display and save metrics.

	Args:
		monitor (Monitor instance, optional): The monitor to run the session on. Defaults to a default Monitor() instance.
	"""
	# monitor.setup_monitoring(agents=[(vendors.QLearningLEAgent, ['QLearning Agent']), (vendors.FixedPriceLEAgent, [(6), 'Rulebased Agent'])], marketplace=sim_market.ClassicScenario)
	if monitor.episodes / monitor.plot_interval > 50:
		print('The ratio of episodes/plot_interval is over 50. In order for the plots to be more readable we recommend a lower ratio.')
		print(f'Episodes: {monitor.episodes}\nPlot Interval: {monitor.plot_interval}\nRatio: {int(monitor.episodes / monitor.plot_interval)}')
		if input('Continue anyway? [y]/n: ') == 'n':
			print('Stopping monitoring session...')
			return

	print('Running a monitoring session with the following configuration:')
	print(str.ljust('Live Drawing enabled:', 25) + str(monitor.enable_live_draw))
	print(str.ljust('Episodes:', 25) + str(monitor.episodes))
	print(str.ljust('Plot interval:', 25) + str(monitor.plot_interval))
	print(str.ljust('Marketplace:', 25) + type(monitor.marketplace).__name__)
	print('Monitoring these agents:')
	for current_agent in monitor.agents:
		print(str.ljust('', 25) + current_agent.name)

	print('\nStarting monitoring session...')
	rewards = monitor.run_marketplace()

	for i in range(len(rewards)):
		print(f'Statistics for agent: {monitor.agents[i].name}')
		print(f'The average reward over {monitor.episodes} episodes is: {np.mean(rewards[i])}')
		print(f'The median reward over {monitor.episodes} episodes is: {np.median(rewards[i])}')
		print(f'The maximum reward over {monitor.episodes} episodes is: {np.max(rewards[i])}')
		print(f'The minimum reward over {monitor.episodes} episodes is: {np.min(rewards[i])}')

	monitor.create_statistics_plots(rewards)
	print(f'All plots were saved to {os.path.abspath(monitor.get_folder())}')


if __name__ == '__main__':  # pragma: no cover
	run_monitoring_session()
