import os
import time

import matplotlib.pyplot as plt

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.market.sim_market as sim_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import CircularAgent, FixedPriceCEAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.market.vendors import Agent, HumanPlayer, RuleBasedAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent


class Configurator():
	"""
	The Configurator is being used together with the `agent_monitoring.Monitor()` and is responsible for managing its configuration.
	"""
	def __init__(self) -> None:
		# Do not change the values in here when setting up a session! They are assumed in tests. Instead use setup_monitoring()!
		ut.ensure_results_folders_exist()
		self.enable_live_draw = False
		self.episodes = 500
		self.plot_interval = 50
		self.marketplace = circular_market.CircularEconomyMonopolyScenario
		default_agent = FixedPriceCEAgent
		self.agents = [default_agent()]
		self.agent_colors = [(0.0, 0.0, 1.0, 1.0)]
		self.folder_path = os.path.abspath(os.path.join(PathManager.results_path, 'monitoring', 'plots_' + time.strftime('%b%d_%H-%M-%S')))

	def get_folder(self) -> str:
		"""
		Return the folder where all diagrams of the current run are saved.

		Returns:
			str: The folder name
		"""
		# create folder with current timestamp to save diagrams at
		os.makedirs(os.path.join(self.folder_path), exist_ok=True)
		return self.folder_path

	def _get_modelfile_path(self, model_name: str) -> str:
		"""
		Get the full path to a modelfile in the 'data' folder.

		Args:
			model_name (str): The name of the .dat modelfile.

		Returns:
			str: The full path to the modelfile.
		"""
		model_name += '.dat'
		full_path = os.path.join(PathManager.data_path, model_name)
		assert os.path.exists(full_path), f'the specified modelfile does not exist: {full_path}'
		return full_path

	def _update_agents(self, agents) -> None:
		"""
		Update the self.agents to the new agents provided.

		Args:
			agents (list of tuples of agent classes and lists): What agents to monitor.
				Must be tuples where the first entry is the class of the agent and the second entry is an optional list
				of arguments for its initialization.
				Each agent will generate data points in the diagrams. See `setup_monitoring()` for more info. Defaults to None.

		Raises:
			RuntimeError: Raised if the modelfile provided does not match the Market/Agent-type provided.
		"""
		# All agents must be of the same type
		assert all(isinstance(agent_tuple, tuple) for agent_tuple in agents), 'agents must be a list of tuples'
		assert all(len(agent_tuple) == 2 for agent_tuple in agents), 'the list entries in agents must have size 2 ([agent_class, arguments])'
		assert all(issubclass(agent_tuple[0], Agent) for agent_tuple in agents), \
			'the first entry in each agent-tuple must be an agent class in `vendors.py`'
		assert all(isinstance(agent_tuple[1], list) for agent_tuple in agents), 'the second entry in each agent-tuple must be a list'
		assert all(issubclass(agent[0], CircularAgent) == issubclass(agents[0][0], CircularAgent) for agent in agents), \
			'the agents must all be of the same type (Linear/Circular)'
		assert issubclass(agents[0][0], CircularAgent) or not isinstance(self.marketplace, circular_market.CircularEconomy), \
			'If the market is circular, the agent must be circular too!'
		assert issubclass(agents[0][0], LinearAgent) or not isinstance(self.marketplace, linear_market.LinearEconomy), \
			'If the market is linear, the agent must be linear too!'

		self.agents = []

		# Instantiate all agents. If they are not rule-based, use the marketplace parameters accordingly
		for current_agent in agents:
			if issubclass(current_agent[0], (RuleBasedAgent, HumanPlayer)):
				# The custom_init takes two parameters: The class of the agent to be initialized and a list of arguments,
				# e.g. for the fixed prices or names
				self.agents.append(Agent.custom_init(current_agent[0], current_agent[1]))
			elif issubclass(current_agent[0], (QLearningAgent, actorcritic_agent.ActorCriticAgent)):
				try:
					assert (0 <= len(current_agent[1]) <= 2), 'the argument list for a RL-agent must have length between 0 and 2'
					assert all(isinstance(argument, str) for argument in current_agent[1]), 'the arguments for a RL-agent must be of type str'

					agent_modelfile = f'{type(self.marketplace).__name__}_{current_agent[0].__name__}'
					agent_name = 'q_learning' if issubclass(current_agent[0], QLearningAgent) else 'actor_critic'
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
					# both arguments, first must be the modelfile, second the name
					elif len(current_agent[1]) == 2:
						assert str.endswith(current_agent[1][0], '.dat'), \
							f'if two arguments are provided, the first one must be the modelfile. Arg1: {current_agent[1][0]}, Arg2: {current_agent[1][1]}'
						agent_modelfile = current_agent[1][0][:-4]
						agent_name = current_agent[1][1]
					# this should never happen due to the asserts before, but you never know
					else:  # pragma: no cover
						raise RuntimeError('invalid arguments provided')

					# create the agent
					new_agent = current_agent[0](marketplace=self.marketplace, load_path=self._get_modelfile_path(agent_modelfile), name=agent_name)
					self.agents.append(new_agent)
				except RuntimeError:  # pragma: no cover
					raise RuntimeError('the modelfile is not compatible with the agent you tried to instantiate')
			else:  # pragma: no cover
				assert False, f'{current_agent[0]} is neither a RuleBased nor a QLearning agent nor a HumanPlayer'

		# set a color for each agent
		color_map = plt.cm.get_cmap('hsv', len(self.agents) + 1)
		self.agent_colors = [color_map(agent_id) for agent_id in range(len(self.agents))]

	def setup_monitoring(
		self,
		enable_live_draw: bool = None,
		episodes: int = None,
		plot_interval: int = None,
		marketplace: sim_market.SimMarket = None,
		agents: list = None,
		subfolder_name: str = None) -> None:
		"""
		Configure the current monitoring session.

		Args:
			enable_live_draw (bool, optional): Whether or not diagrams should be displayed on screen when drawn. Defaults to None.
			episodes (int, optional): The number of episodes to run. Defaults to None.
			plot_interval (int, optional): After how many episodes a new data point/plot should be generated. Defaults to None.
			marketplace (sim_market class, optional): What marketplace to run the monitoring on. Defaults to None.
			agents (list of tuples of agent classes and lists): What agents to monitor. Each entry must be a tuple of a valid agent class and a list
				of optional arguments, where a .dat modelfile and/or a name for the agent can be specified.
				Modelfile defaults to \'marketplaceClass_AgentClass.dat\', Name defaults to \'q_learning\'
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
			assert plot_interval <= self.episodes, \
				f'plot_interval must be <= episodes, or no plots can be generated. Episodes: {self.episodes}. Plot_interval: {plot_interval}'
			self.plot_interval = plot_interval

		if(marketplace is not None):
			assert issubclass(marketplace, sim_market.SimMarket), 'the marketplace must be a subclass of SimMarket'
			self.marketplace = marketplace()
			# If the agents have not been changed, we reuse the old agents
			if(agents is None):
				print('Warning: Your agents are being overwritten by new instances of themselves!')
				agents = [(type(current_agent), []) for current_agent in self.agents]
			self._update_agents(agents)

		# marketplace has not changed but agents have
		elif(agents is not None):
			self._update_agents(agents)

		if(subfolder_name is not None):
			assert isinstance(subfolder_name, str), f'subfolder_name must be of type str: {type(subfolder_name)}, {subfolder_name}'
			self.folder_path = os.path.join(PathManager.results_path, 'monitoring', subfolder_name)

	def print_configuration(self):
		"""
		Print the current configuration in a human-readable format.

		Used when running a monitoring session from `agent_monitoring.Monitor()`.
		"""
		if self.episodes / self.plot_interval > 50:
			print('The ratio of episodes/plot_interval is over 50. In order for the plots to be more readable we recommend a lower ratio.')
			print(f'Episodes: {self.episodes}')
			print(f'Plot Interval: {self.plot_interval}')
			print(f'Ratio: {int(self.episodes / self.plot_interval)}')
			if input('Continue anyway? [y]/n: ') == 'n':
				print('Stopping monitoring session...')
				return

		print('Running a monitoring session with the following configuration:')
		print(str.ljust('Live Drawing enabled:', 25) + str(self.enable_live_draw))
		print(str.ljust('Episodes:', 25) + str(self.episodes))
		print(str.ljust('Plot interval:', 25) + str(self.plot_interval))
		print(str.ljust('Marketplace:', 25) + type(self.marketplace).__name__)
		print('Monitoring these agents:')
		for current_agent in self.agents:
			print(str.ljust('', 25) + current_agent.name)


if __name__ == '__main__':  # pragma: no cover
	raise RuntimeError('agent_monitoring can only be run from `am_monitoring.py`')
