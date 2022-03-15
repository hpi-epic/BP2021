# helper
import importlib
import json
import os
from abc import ABC, abstractmethod

from alpha_business.agents.vendors import CircularAgent, QLearningAgent
from alpha_business.market.circular.circular_sim_market import CircularEconomy
from alpha_business.market.sim_market import SimMarket
from alpha_business.rl.actorcritic_agent import ActorCriticAgent


class EnvironmentConfig(ABC):
	"""
	An abstract environment configuration class.
	"""

	def __init__(self, config: dict):
		self.task = self._get_task()
		self._validate_config(config)

	def __str__(self) -> str:
		"""
		This overwrites the internal function that get called when you call `print(class_instance)`.

		Instead of printing the class name, prints the instance variables as a dictionary.

		Returns:
			str: The instance variables as a dictionary.
		"""
		return f'{self.__class__.__name__}: {self.__dict__}'

	def _validate_config(self, config: dict, single_agent: bool, needs_modelfile: bool) -> None:
		"""
		Validate the given configuration dictionary and set the instance variables accordingly.

		Args:
			config (dict): The config dictionary to be validated.
			single_agent (bool): Whether or not only one agent should be used.
				Note that if single_agent is True and the agent dictionary is too long, it will be shortened globally.
			needs_modelfile (bool): Whether or not the config must include modelfiles.

		Raises:
			AssertionError: In case the provided configuration is invalid.
		"""

		# CHECK: All required top-level fields exist
		assert 'agents' in config, f'The config must have an "agents" field: {config}'
		assert 'marketplace' in config, f'The config must have a "marketplace" field: {config}'

		# CHECK: Marketplace
		assert isinstance(config['marketplace'], str), \
			f'The "marketplace" field must be a str: {config["marketplace"]} ({type(config["marketplace"])})'

		self.marketplace = self._get_class(config['marketplace'])
		assert issubclass(self.marketplace, SimMarket), f'The marketplace passed must be a subclass of SimMarket: {self.marketplace}'

		# CHECK: Agents
		assert isinstance(config['agents'], dict), \
			f'The "agents" field must be a dict: {config["agents"]} ({type(config["agents"])})'

		# Shorten the agent dictionary if only one is necessary
		if single_agent and len(config['agents']) > 1:
			used_agent = list(config['agents'].items())[0]
			config['agents'] = {used_agent[0]: used_agent[1]}
			print(f'Multiple agents were provided but only the first one will be used:\n{config["agents"]}\n')

		# Save the agents in config['agents'] in a list for easier access
		agent_dictionaries = [config['agents'][agent] for agent in config['agents']]

		assert (isinstance(agent, dict) for agent in agent_dictionaries), \
			f'All agents in the "agents" field must be dictionaries: {[config["agents"][agent] for agent in config["agents"]]}'

		# CHECK: Agents::Class
		assert all('class' in agent for agent in agent_dictionaries), f'Each agent must have a "class" field: {agent_dictionaries}'
		assert all(isinstance(agent['class'], str) for agent in agent_dictionaries), \
			f'The "class" fields must be strings: {agent_dictionaries} ({[type(agent["class"]) for agent in agent_dictionaries]})'

		# CHECK: Agents::Modelfile
		agent_classes = [self._get_class(agent['class']) for agent in agent_dictionaries]
		# If a modelfile is needed, the self.agents will be a list of tuples (as required by agent_monitoring), else just a list of classes
		if needs_modelfile:
			modelfile_list = []
			for current_agent in range(len(agent_classes)):
				if issubclass(agent_classes[current_agent], (QLearningAgent, ActorCriticAgent)):
					assert 'modelfile' in agent_dictionaries[current_agent], f'This agent must have a "modelfile" field: {agent_classes[current_agent]}'

					modelfile = agent_dictionaries[current_agent]['modelfile']

					assert isinstance(modelfile, str), \
						f'The "modelfile" field of this agent must be a str: {agent_classes[current_agent]} ({type(modelfile)})'
					# Check that the modelfile exists. Implies that it must end in .dat Taken from am_configuration::_get_modelfile_path()
					full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'data', modelfile))
					assert os.path.exists(full_path), f'the specified modelfile does not exist: {full_path}'
					modelfile_list.append(modelfile)

				# if this agent doesn't have modelfiles, append None
				# we need to append *something* since the subsequent call creates a list of tuples using the `modelfile_list`
				# if we were to only append items for agents with modelfiles, the lists would have different lengths and the
				# process of matching the correct ones would get a lot more difficult
				else:
					modelfile_list.append(None)
			# Create a list of tuples (agent_class, modelfile_string)
			self.agent = list(zip(agent_classes, iter(modelfile_list)))

			assert all(issubclass(agent[0], CircularAgent) == issubclass(self.marketplace, CircularEconomy) for agent in self.agent), \
				f'The agents and marketplace must be of the same economy type (Linear/Circular): {self.agent} and {self.marketplace}'
		else:
			self.agent = agent_classes
			assert all(issubclass(agent, CircularAgent) == issubclass(self.marketplace, CircularEconomy) for agent in self.agent), \
				f'The agents and marketplace must be of the same economy type (Linear/Circular): {self.agent} and {self.marketplace}'

		# If only one agent is needed, we just use the first agent from the list we created before
		if single_agent:
			self.agent = self.agent[0]

	def _get_class(self, import_string: str) -> object:
		"""
		Get the class from the given string.

		Args:
			import_string (str): A string containing the import path in the format 'module.submodule.class'.

		Returns:
			A class object: The imported class.
		"""
		module_name, class_name = import_string.rsplit('.', 1)
		try:
			return getattr(importlib.import_module(module_name), class_name)
		except AttributeError as error:
			raise AttributeError(f'The string you passed could not be resolved to a class: {import_string}') from error
		except ModuleNotFoundError as error:
			raise ModuleNotFoundError(f'The string you passed could not be resolved to a module: {import_string}') from error

	@abstractmethod
	def _get_task(self) -> str:
		"""
		Return the type of task this Config is for.

		Returns:
			str: The task name.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')


class TrainingEnvironmentConfig(EnvironmentConfig):
	"""
	The environment configuration class for the training configuration.

	Instance variables:
		task (str): The task this config can be used for. Always "training".
		marketplace (SimMarket subclass): A subclass of SimMarket, what marketplace the training should be run on.
		agent (QlearningAgent or ActorCriticAgent subclass): A subclass of QlearningAgent or ActorCritic, the agent to be trained.
	"""

	def _validate_config(self, config: dict) -> None:
		super(TrainingEnvironmentConfig, self)._validate_config(config, single_agent=True, needs_modelfile=False)

		assert issubclass(self.agent, (QLearningAgent, ActorCriticAgent)), \
			f'The agent class passed must be subclasses of either QLearningAgent or ActorCriticAgent: {self.agent}'

	def _get_task(self) -> str:
		return 'training'


class AgentMonitoringEnvironmentConfig(EnvironmentConfig):
	"""
	The environment configuration class for the agent_monitoring configuration.

	Instance variables:
		task (str): The task this config can be used for. Always "agent_monitoring".
		enable_live_draw (bool): Whether or not live drawing should be enabled.
		episodes (int): The number of episodes to run the monitoring for.
		plot_interval (int): The interval between plot creation.
		marketplace (SimMarket subclass): A subclass of SimMarket, what marketplace the monitoring session should be run on.
		agent (list of tuples): A list containing the agents that should be trained.
			Each entry in the list is a tuple with the first item being the agent class, the second being a list.
			If the agent needs a modelfile, this will be the first entry in the list, the other entry is always an informal name for the agent.
	"""

	def _validate_config(self, config: dict) -> None:
		# TODO: subfolder_name variable
		# CHECK: All required top-level fields exist
		assert 'enable_live_draw' in config, f'The config must have an "enable_live_draw" field: {config}'
		assert 'episodes' in config, f'The config must have an "episodes" field: {config}'
		assert 'plot_interval' in config, f'The config must have an "plot_interval" field: {config}'

		# CHECK: Agent_monitoring fields have the correct types
		assert isinstance(config['enable_live_draw'], bool), \
			f'The "enable_live_draw" field must be a bool: {config["enable_live_draw"]} ({type(config["enable_live_draw"])})'
		assert isinstance(config['episodes'], int), \
			f'The "episodes" field must be a int: {config["episodes"]} ({type(config["episodes"])})'
		assert isinstance(config['plot_interval'], int), \
			f'The "plot_interval" field must be a int: {config["plot_interval"]} ({type(config["plot_interval"])})'

		self.enable_live_draw = config['enable_live_draw']
		self.episodes = config['episodes']
		self.plot_interval = config['plot_interval']

		# We do the super call last because getting the classes takes longer than the other operations, so we save time in case of an error.
		super(AgentMonitoringEnvironmentConfig, self)._validate_config(
			config, single_agent=False, needs_modelfile=True)

		# Since RuleBasedAgents do not have modelfiles, we need to adjust the passed lists to remove the "None" entry
		passed_agents = self.agent
		self.agent = []
		for current_agent in range(len(passed_agents)):
			# No modelfile
			if passed_agents[current_agent][1] is None:
				self.agent.append((passed_agents[current_agent][0], [list(config['agents'].keys())[current_agent]]))
			else:
				self.agent.append((passed_agents[current_agent][0], [passed_agents[current_agent][1], list(config['agents'].keys())[current_agent]]))

	def _get_task(self) -> str:
		return 'agent_monitoring'


class ExampleprinterEnvironmentConfig(EnvironmentConfig):
	"""
	The environment configuration class for the exampleprinter configuration.

	Instance variables:
		task (str): The task this config can be used for. Always "exampleprinter".
		marketplace (SimMarket subclass): A subclass of SimMarket, what marketplace the exampleprinter should be run on.
		agent (Agent subclass): A subclass of Agent, the agent for which the exampleprinter should be run.
	"""

	def _validate_config(self, config: dict) -> None:
		super(ExampleprinterEnvironmentConfig, self)._validate_config(config, single_agent=True, needs_modelfile=True)

	def _get_task(self) -> str:
		return 'exampleprinter'


class EnvironmentConfigLoader():
	"""
	This class is used to load a json-file containing a generic configuration and instantiate the correct
	`EnvironmentConfig` object to pass to e.g. the `training_scenario.py`.
	It can also be used to simply validate an existing dictionary containing a configuration.
	"""

	def load(filename: str) -> EnvironmentConfig:
		"""
		Load the configuration json file from the specified path and instantiate the correct configuration class.

		Args:
			filename (str): The name of the json file containing the configuration values.
				Must be located in the BP2021/ folder.

		Returns:
			EnvironmentConfig: A subclass instance of EnvironmentConfig.
		"""
		filename += '.json'
		path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, filename)
		with open(path) as config_file:
			config = json.load(config_file)
		return EnvironmentConfigLoader.validate(config)

	def validate(config: dict) -> EnvironmentConfig:
		"""
		Validate the given config dictionary and return the correct configuration class.

		Args:
			config (dict): The configuration to validate.

		Raises:
			AssertionError: If the given configuration has an unknown task name.

		Returns:
			EnvironmentConfig: A subclass instance of EnvironmentConfig.
		"""
		if config['task'] == 'training':
			return TrainingEnvironmentConfig(config)
		elif config['task'] == 'agent_monitoring':
			return AgentMonitoringEnvironmentConfig(config)
		elif config['task'] == 'exampleprinter':
			return ExampleprinterEnvironmentConfig(config)
		else:
			raise AssertionError(f'The specified task is unknown: {config["task"]}\nConfig: {config}')

	def is_valid(config: dict):
		"""
		To be used when the actual config object is not necessary but only validity needs to be checked.
		Validates a given config and catches any possible exceptions.
		Returns if a config is valid and if not, also an appropriate error message.

		Args:
			config (dict): The configuration to validate.

		Returns:
			Tuple (bool, str): boolean indicating if your config is valid, str the appropriate error if applicable.
		"""
		try:
			EnvironmentConfigLoader.validate(config)
		except (AssertionError, Exception) as error:
			return False, str(error)
		return True, 'Your config is valid.'


if __name__ == '__main__':  # pragma: no cover
	config: ExampleprinterEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_exampleprinter')
	print(config)
	print()
	config: AgentMonitoringEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_agent_monitoring')
	print(config)
	print()
	config: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	print(config)
