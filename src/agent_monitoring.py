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
		self.enable_live_draws = True
		self.episodes = 200
		self.plot_interval = int(self.episodes / 10)
		# should get deprecated when introducing possibility to use multiple RL-agents
		self.path_to_modelfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + 'QLearningAgent_ClassicScenario.dat'
		self.marketplace = sim_market.ClassicScenario()
		self.agents = [agent.QLearningAgent(self.marketplace.observation_space.shape[0], self.marketplace.action_space.n, load_path=self.path_to_modelfile)]
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

	def update_agents(self, agents) -> None:
		"""
		Update the self.agents to the new agens provided

		Args:
			agents (list of agent classes, optional): What agents to monitor. Each agent will generate data points in the diagrams. Defaults to None.

		Raises:
			RuntimeError: Raised if the modelfile provided does not match the Market/Agent-type provided
		"""
		n_actions = 1
		if isinstance(self.marketplace.action_space, gym.spaces.Discrete):
			n_actions = self.marketplace.action_space.n
		else:
			for id in range(0, len(self.marketplace.action_space)):
				n_actions *= self.marketplace.action_space[id].n

		# All agents must be of the same type
		assert all(agent_class.is_circular == agents[0].is_circular for agent_class in agents), 'The agents are not of the same type!'
		self.agents = []

		# Instantiate all agents. If they are not rule-based, use the marketplace parameters accordingly
		for agent_class in agents:
			if agent_class.is_rule_based:
				self.agents.append(agent_class())
			elif not agent_class.is_rule_based:
				# TODO: Modelfile from list!
				try:
					self.agents.append(agent_class(self.marketplace.observation_space.shape[0], n_actions, load_path=self.path_to_modelfile))
				except RuntimeError:
					raise RuntimeError('The modelfile is probably not compatible with the agent you tried to instantiate!')
			else:
				assert False, agent_class + 'is neither a rule_based nor a reinforcement_learning agent!'

		# set a color for each agent
		color_map = self.get_cmap(len(self.agents))
		self.agent_colors = []
		for agent_id in range(0, len(self.agents)):
			self.agent_colors.append(color_map(agent_id))

	def setup_monitoring(self, draw_enabled=None, episodes=None, plot_interval=None, modelfile=None, agents=None, marketplace=None, subfolder_name=None) -> None:
		"""
		Configure the current monitoring session.

		Args:
			draw_enabled (bool, optional): Whether or not diagrams should be displayed on screen when drawn. Defaults to None.
			episodes (int, optional): The number of episodes to run. Defaults to None.
			plot_interval (int, optional): After how many episodes a new data point/plot should be generated. Defaults to None.
			modelfile (str, optional): Path to the file containing the model for a RL-agent. Defaults to None.
			agents (list of agent classes, optional): What agents to monitor. Each agent will generate data points in the diagrams. Defaults to None.
			marketplace (sim_market instance, optional): What marketplace to run the monitoring on. Defaults to None.
			subfolder_name (str, optional): The name of the folder to save the diagrams in. Defaults to None.
		"""
		# doesn't look nice, but afaik only way to keep parameter list short
		if(draw_enabled is not None):
			self.enable_live_draws = draw_enabled
		if(episodes is not None):
			self.episodes = episodes
		if(plot_interval is not None):
			self.plot_interval = plot_interval
		if(modelfile is not None):
			self.path_to_modelfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + modelfile

		if(marketplace is not None):
			self.marketplace = marketplace
			# The agents have not been changed, we reuse the old agents
			if(agents is None):
				# A bit hacky
				agents = self.agents
			self.update_agents(agents)

		# marketplace has not changed but agents have
		elif(agents is not None):
			self.update_agents(agents)

		if(subfolder_name is not None):
			self.subfolder_name = subfolder_name
			self.folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + self.subfolder_name

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

		if self.enable_live_draws:  # pragma: no cover
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
		metrics_functions = [self.metrics_average, self.metrics_maximum, self.metrics_median, self.metrics_minimum]
		metrics_names = ['Average', 'Maximum', 'Median', 'Minimum']
		x_axis_episodes = np.arange(self.plot_interval, self.episodes + 1, self.plot_interval)

		for function in range(len(metrics_functions)):
			# calculate <metric> rewards per self.plot_interval episodes for each agent
			metric_rewards = []
			for agent_rewards_id in range(0, len(rewards)):
				metric_rewards.append([])
				for starting_index in range(0, int(len(rewards[agent_rewards_id]) / self.plot_interval)):
					metric_rewards[agent_rewards_id].append(metrics_functions[function](rewards[agent_rewards_id][0:self.plot_interval * starting_index + self.plot_interval]))
			self.create_line_plot(x_axis_episodes, metric_rewards, metrics_names[function])

	def create_step_plots(self, all_steps_rewards) -> None:
		"""
		Create a reward per step plot for all agents.

		Args:
			all_steps_rewards (list of list of float): A list containing a list for each agent, in these lists there are the rewards for this agent per step.
		"""
		# this is what we used to log in the first place
		episode_rewards = self.get_episode_rewards(all_steps_rewards)
		# average over episode
		mean_episode_rewards = []
		for agent_reward in episode_rewards:
			mean_episode_rewards.append([])
			agent_reward = [reward / ut.EPISODE_LENGTH for reward in agent_reward]
			mean_episode_rewards[-1] = agent_reward

		mean_episode_rewards = [reward[::int(self.episodes / self.plot_interval)] for reward in mean_episode_rewards]
		steps = range(0, int(self.episodes))[::int(self.episodes / self.plot_interval)]
		self.create_line_plot(steps, mean_episode_rewards, 'average step')

	def create_line_plot(self, episodes, metric_rewards, metric_name='default') -> None:
		"""Create a line plot with the given rewards data.

		Args:
			episodes (array of ints): Defines x-values of datapoints. Must have same length as metric_rewards.
			metric_rewards (array of array of ints): Defines y-values of datapoints, one array per monitored agent. Must have same length as episodes.
			metric_name (str, optional): Used for naming the y-axis, diagram and output file. Defaults to 'default'.
		"""
		print(f'Creating line plot for {metric_name} rewards...')
		# clear old plot completely
		plt.clf()
		filename = metric_name + '_rewards.svg'
		# plot the metric rewards for each agent
		for index in range(len(metric_rewards)):
			plt.plot(episodes, metric_rewards[index], marker='o', color=self.agent_colors[index])

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

		Returns:
			list: A list with a list of rewards for each agent
		"""

		# initialize the rewards list with a list for each agent
		rewards = []
		for i in range(len(self.agents)):
			rewards.append([])

		all_steps_rewards = []
		for i in range(len(self.agents)):
			all_steps_rewards.append([])

		for episode in range(1, self.episodes + 1):
			# reset the state once to be used by all agents
			default_state = self.marketplace.reset()

			for i in range(0, len(self.agents)):
				# reset marketplace, bit hacky, if you find a better solution feel free
				self.marketplace.reset()
				self.marketplace.state = default_state

				# reset values for all agents
				state = default_state
				# episode_reward = 0
				is_done = False

				# run marketplace for this agent
				while not is_done:
					action = self.agents[i].policy(state)
					state, step_reward, is_done, _ = self.marketplace.step(action)
					# episode_reward += step_reward
					# this gives us a higher flexibility in terms of what metrics we would like to use
					all_steps_rewards[i] += [step_reward]

				# removing this will decrease our performance when we still want to do live drawing
				# could think about a caching strategy for live drawing
				# add the reward to the current agent's reward-Array
				# rewards[i] += [episode_reward]

			# after all agents have run the episode
			if (episode % 100) == 0:
				print(f'Running {episode}th episode...')

			if (episode % self.plot_interval) == 0:
				self.create_histogram(self.get_episode_rewards(all_steps_rewards), 'episode_' + str(episode))
		return all_steps_rewards


monitor = Monitor()


def main() -> None:
	import agent
	monitor.setup_monitoring(modelfile='QLearningCEAgent_CircularEconomy.dat', marketplace=sim_market.CircularEconomy(), agents=[agent.RuleBasedCEAgent, agent.QLearningCEAgent])
	print(f'Running', monitor.episodes, 'episodes')
	print(f'Plot interval is: {monitor.plot_interval}')
	print(f'Using modelfile: {monitor.path_to_modelfile}')
	print(f'The marketplace is: {monitor.marketplace}')
	print('Monitoring these agents:')
	for current_agent in monitor.agents:
		print(current_agent.name)

	all_steps_rewards = monitor.run_marketplace()

	rewards = monitor.get_episode_rewards(all_steps_rewards)
	for i in range(len(rewards)):
		print(f'Statistics for agent: {monitor.agents[i].name}')
		print(f'The average reward over {monitor.episodes} episodes is: {str(monitor.metrics_average(rewards[i]))}')
		print(f'The median reward over {monitor.episodes} episodes is: {str(monitor.metrics_median(rewards[i]))}')
		print(f'The maximum reward over {monitor.episodes} episodes is: {str(monitor.metrics_maximum(rewards[i]))}')
		print(f'The minimum reward over {monitor.episodes} episodes is: {str(monitor.metrics_minimum(rewards[i]))}')

	monitor.create_step_plots(all_steps_rewards)
	monitor.create_statistics_plots(rewards)
	print(f'All plots were saved to {monitor.get_folder()}')


if __name__ == '__main__':  # pragma: no cover
	main()
