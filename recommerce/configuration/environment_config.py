# helper
import importlib
import json
import os
from abc import ABC, abstractmethod

from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomy
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import FixedPriceAgent
from recommerce.rl.actorcritic.actorcritic_agent import ActorCriticAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent


def get_class(import_string: str) -> object:
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


class EnvironmentConfig(ABC):
	"""
	An abstract environment configuration class.
	"""

	def __init__(self, config: dict):
		self.task = self._get_task()
		self._validate_config(config)

	@classmethod
	def get_required_fields(cls, dict_key) -> dict:
		"""
		Utility function that returns all of the keys required for a environment_config.json at the given level.
		The value of any given key indicates whether or not it is the key of a dictionary within the config (i.e. they are a level themselves).

		Args:
			dict_key (str): The key for which the required fields are needed. 'top-dict' for getting the keys of the first level.

		Returns:
			dict: The required keys for the config at the given level, together with a boolean indicating of they are the key
				of another level.

		Raises:
			AssertionError: If the given level is invalid.
		"""
		if dict_key == 'top-dict':
			return {
				'task': False,
				'enable_live_draw': False,
				'episodes': False,
				'plot_interval': False,
				'marketplace': False,
				'agents': True
			}
		elif dict_key == 'agents':
			return {
				'agent_class': False,
				'argument': False
			}
		else:
			raise AssertionError(f'The given level does not exist in an environment-config: {dict_key}')

	# This function should always contain ALL keys that are possible, so the webserver-config is independent of the given "task"
	# since the user does not need to specify a "task". The subclasses should overwrite this method.
	@classmethod
	def check_types(cls, config: dict, task: str = 'None', must_contain: bool = True) -> None:
		"""
		Check if all given variables have the correct types.
		If must_contain is True, all keys must exist, else non-existing keys will be skipped.

		Args:
			config (dict): The config to check.
			task (str): The task for which the variables should be checked.
			must_contain (bool, optional): Whether or not all variables must be present in the config. Defaults to True.

		Raises:
			AssertionError: If an unknown key was passed.
			KeyError: If the dictionary is missing a key but should contain all keys.
			ValueError: If one of the passed strings (marketplace, agents) could not be parsed to a valid class.
		"""
		if task in {'None', 'agent_monitoring'}:
			types_dict = {
				'task': str,
				'enable_live_draw': bool,
				'episodes': int,
				'plot_interval': int,
				'marketplace': str,
				'agents': dict
			}
			types_dict_agents = {
				'agent_class': str,
				# str for modelfiles, list for FixedPrice-Agent price-list
				'argument': (str, list)
			}

		elif task in {'training', 'exampleprinter'}:
			types_dict = {
				'task': str,
				'marketplace': str,
				'agents': dict
			}
			types_dict_agents = {
				'agent_class': str,
				# str for modelfiles, list for FixedPrice-Agent price-list
				'argument': (str, list)
			}
		else:
			raise AssertionError(f'This task is unknown: {task}')

		for key, value in types_dict.items():
			try:
				assert isinstance(config[key], value), f'{key} must be a {value} but was {type(config[key])}'
				# make sure the class can be parsed/is valid
				if key == 'marketplace':
					try:
						get_class(config['marketplace'])
					except ValueError as error:
						raise AssertionError(f'The marketplace could not be parsed to a valid class: "{config["marketplace"]}"') from error
				# TODO: Refactor this when the agent structure was changed in the json files
				if key == 'agents':
					for agent in config['agents']:
						for agent_key, agent_value in types_dict_agents.items():
							if agent_key == 'agent_class':
								try:
									get_class(config['agents'][agent]['agent_class'])
								except ValueError as error:
									raise AssertionError(f'This agent could not be parsed to a valid class: \
"{config["agents"][agent]["agent_class"]}"') from error
							assert isinstance(config['agents'][agent][agent_key], agent_value), \
								f'{agent_key} must be a {agent_value} but was {type(config["agents"][agent][agent_key])}'
			except KeyError as error:
				if must_contain:
					raise KeyError(f'Your config is missing the following required key: {key}') from error

	def __str__(self) -> str:
		"""
		This overwrites the internal function that get called when you call `print(class_instance)`.

		Instead of printing the class name, prints the instance variables as a dictionary.

		Returns:
			str: The instance variables as a dictionary.
		"""
		return f'{self.__class__.__name__}: {self.__dict__}'

	def _check_top_level_structure(self, config: dict) -> None:
		"""
		Utility function that checks if all required top-level fields exist and have the right types.

		Args:
			config (dict): The config to be checked.
		"""
		assert 'task' in config, f'The config must have a "task" field: {config}'
		assert 'marketplace' in config, f'The config must have a "marketplace" field: {config}'
		assert 'agents' in config, f'The config must have an "agents" field: {config}'

		self.check_types(config, config['task'])

	def _check_and_adjust_agents_structure(self, agents_config: dict, single_agent: bool) -> tuple:
		"""
		Utility function that checks if the agents field has the correct structure and shortens it if necessary.

		Args:
			agents_config (dict): The dict to be checked.
			single_agent (bool): Whether or not only one agent should be used.
				Note that if single_agent is True and the agent dictionary is too long, it will be shortened globally.

		Returns:
			dict: The agents_config, shortened if single_agent is True.
			list: The list of agent_dictionaries extracted from agents_config.
		"""
		assert all(isinstance(agents_config[agent], dict) for agent in agents_config), \
			f'All agents in the "agents" field must be dictionaries: {[agents_config[agent] for agent in agents_config]}, \
{[type(agents_config[agent]) for agent in agents_config]}'

		# Shorten the agent dictionary if only one is necessary
		if single_agent and len(agents_config) > 1:
			used_agent = list(agents_config.items())[0]
			agents_config = {used_agent[0]: used_agent[1]}
			print(f'Multiple agents were provided but only the first one will be used:\n{agents_config}\n')

		# Save the agents in agents_config in a list for easier access
		agent_dictionaries = [agents_config[agent] for agent in agents_config]

		# CHECK: Agents::agent_class
		assert all('agent_class' in agent for agent in agent_dictionaries), f'Each agent must have an "agent_class" field: {agent_dictionaries}'
		assert all(isinstance(agent['agent_class'], str) for agent in agent_dictionaries), \
			f'The "agent_class" fields must be strings: {agent_dictionaries} ({[type(agent["agent_class"]) for agent in agent_dictionaries]})'

		# CHECK: Agents::argument
		assert all('argument' in agent for agent in agent_dictionaries), f'Each agent must have an "argument" field: {agent_dictionaries}'

		return agents_config, agent_dictionaries

	def _parse_agent_arguments(self, agent_dictionaries: dict, needs_modelfile: bool) -> tuple:
		"""
		Utility function that parses the provided agent arguments, making sure they are the correct type for the agent.

		Args:
			agent_dictionaries (dict): The agents for which to parse the arguments.
			needs_modelfile (bool): Whether or not RL-agents need modelfiles in this config.

		Returns:
			list: A list of agent classes.
			list: A list of parsed arguments.
		"""
		agent_classes = [get_class(agent['agent_class']) for agent in agent_dictionaries]

		# If a modelfile is needed, the self.agents will be a list of tuples (as required by agent_monitoring), else just a list of classes
		arguments_list = []
		for current_agent in range(len(agent_classes)):
			current_config_argument = agent_dictionaries[current_agent]['argument']

			# This if-else contains the parsing logic for the different types of arguments agents can have, e.g. modelfiles or fixed-price-lists
			if needs_modelfile and issubclass(agent_classes[current_agent], (QLearningAgent, ActorCriticAgent)):
				assert isinstance(current_config_argument, str), \
					f'The "argument" field of this agent must be a string: {agent_classes[current_agent]} ({type(current_config_argument)})'
				assert current_config_argument.endswith('.dat'), \
					f'The "argument" field must be a modelfile and therefore end in ".dat": {current_config_argument}'
				# Check that the modelfile exists. Taken from am_configuration::_get_modelfile_path()
				full_path = os.path.abspath(os.path.join(PathManager.data_path, current_config_argument))
				assert os.path.exists(full_path), f'the specified modelfile does not exist: {full_path}'

				arguments_list.append(current_config_argument)

			elif issubclass(agent_classes[current_agent], FixedPriceAgent):
				assert isinstance(current_config_argument, list), \
					f'The "argument" field of this agent must be a list: {agent_classes[current_agent]} ({type(current_config_argument)})'
				# Subclasses of FixedPriceAgent solely accept tuples
				arguments_list.append(tuple(current_config_argument))

			# if this agent doesn't have modelfiles or *fixed_price-lists*, append None
			# we need to append *something* since the subsequent call creates a list of tuples using the `arguments_list`
			# if we were to only append items for agents with modelfiles or *fixed_price-lists*, the lists would have different lengths and the
			# process of matching the correct ones would get a lot more difficult
			else:
				if current_config_argument != '' and current_config_argument is not None:
					print(f'Your passed argument {current_config_argument} in the "argument" field will be discarded!')
				arguments_list.append(None)

		return agent_classes, arguments_list

	def _set_marketplace(self, marketplace_string: str) -> None:
		"""
		Utility function that validates the type of marketplace passed and sets the instance variable.

		Args:
			marketplace (str): The string of the class within the config dictionary.
		"""
		self.marketplace = get_class(marketplace_string)
		assert issubclass(self.marketplace, SimMarket), \
			f'The type of the passed marketplace must be a subclass of SimMarket: {self.marketplace}'

	def _set_agents(self, agent_classes: list, arguments_list: list) -> None:
		"""
		Utility function that creates a list of tuples from the agent classes and their arguments
		and sets the resulting list as an instance variable.

		Args:
			agent_classes (list): A list of the different agent classes.
			arguments_list (list): A list of arguments for the different agents.
		"""
		# Create a list of tuples (agent_class, argument)
		self.agent = list(zip(agent_classes, arguments_list))

	def _assert_agent_marketplace_fit(self) -> None:
		"""
		Utility function that makes sure the agent(s) and marketplace are of the same type.
		"""

		assert all(issubclass(agent[0], CircularAgent) == issubclass(self.marketplace, CircularEconomy) for agent in self.agent), \
			f'The agents and marketplace must be of the same economy type (Linear/Circular): {self.agent} and {self.marketplace}'

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

		self._check_top_level_structure(config)

		config['agents'], agent_dictionaries = self._check_and_adjust_agents_structure(config['agents'], single_agent)

		agent_classes, arguments_list = self._parse_agent_arguments(agent_dictionaries, needs_modelfile)

		self._set_agents(agent_classes, arguments_list)

		self._set_marketplace(config['marketplace'])

		self._assert_agent_marketplace_fit()

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

		# Since we only have one agent without any arguments, we extract it from the provided list
		self.agent = self.agent[0][0]
		assert issubclass(self.agent, (QLearningAgent, ActorCriticAgent)), \
			f'The first component must be a subclass of either QLearningAgent or ActorCriticAgent: {self.agent}'

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
	def _check_top_level_structure(self, config: dict) -> None:
		super(AgentMonitoringEnvironmentConfig, self)._check_top_level_structure(config)
		assert 'enable_live_draw' in config, f'The config must have an "enable_live_draw" field: {config}'
		assert 'episodes' in config, f'The config must have an "episodes" field: {config}'
		assert 'plot_interval' in config, f'The config must have a "plot_interval" field: {config}'

		assert isinstance(config['enable_live_draw'], bool), \
			f'The "enable_live_draw" field must be a bool: {config["enable_live_draw"]} ({type(config["enable_live_draw"])})'
		assert isinstance(config['episodes'], int), \
			f'The "episodes" field must be a int: {config["episodes"]} ({type(config["episodes"])})'
		assert isinstance(config['plot_interval'], int), \
			f'The "plot_interval" field must be a int: {config["plot_interval"]} ({type(config["plot_interval"])})'

	def _validate_config(self, config: dict) -> None:
		# TODO: subfolder_name variable

		super(AgentMonitoringEnvironmentConfig, self)._validate_config(config, single_agent=False, needs_modelfile=True)

		self.enable_live_draw = config['enable_live_draw']
		self.episodes = config['episodes']
		self.plot_interval = config['plot_interval']

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
		# Since we only have one agent, we extract it from the provided list
		self.agent = self.agent[0]

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
				Must be located in the user's datapath folder.

		Returns:
			EnvironmentConfig: A subclass instance of EnvironmentConfig.
		"""
		filename += '.json'
		path = os.path.join(PathManager.user_path, filename)
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
		assert 'task' in config, f'The config must have a "task" field: {config}'
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
