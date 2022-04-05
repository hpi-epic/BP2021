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
				'agents': False
			}
		elif dict_key == 'agents':
			return {
				'agent_class': False,
				'argument': False
			}
		else:
			raise AssertionError(f'The given level does not exist in an environment-config: {dict_key}')

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
				'agents': list
			}
		elif task in {'training', 'exampleprinter'}:
			types_dict = {
				'task': str,
				'marketplace': str,
				'agents': list
			}
		else:
			raise AssertionError(f'This task is unknown: {task}')

		types_dict_agents = {
			'name': str,
			'agent_class': str,
			# str for modelfiles, list for FixedPrice-Agent price-lists
			'argument': (str, list)
		}

		for key, value in types_dict.items():
			try:
				assert isinstance(config[key], value), f'{key} must be a {value} but was {type(config[key])}'
				# make sure the agent-classes can be parsed and that each entry in the dictionary has the correct type
				if key == 'agents':
					for agent in config['agents']:
						# check types of the entries in the current agent dictionary
						for checked_key, checked_value in types_dict_agents.items():
							assert isinstance(agent[checked_key], checked_value), \
								f'{checked_key} must be a {checked_value} but was {type(agent[checked_key])}'
						try:
							get_class(agent['agent_class'])
						except ValueError as error:
							raise AssertionError(f'This agent could not be parsed to a valid class: "{config["agents"][agent]["agent_class"]}"') from error
				# make sure the marketplace class can be parsed/is valid
				elif key == 'marketplace':
					try:
						get_class(config['marketplace'])
					except Exception as error:
						raise AssertionError(f'The marketplace could not be parsed to a valid class: "{config["marketplace"]}"') from error
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

	def _check_config_structure(self, config: dict, single_agent: bool) -> None:
		"""
		Utility function that checks if all required fields exist and have the right types.

		Args:
			config (dict): The config to be checked.
			single_agent (bool): Whether or not only one agent should be used.
		"""
		assert 'task' in config, f'The config must have a "task" field: {config}'

		if single_agent:
			# we check the agents-type prematurely to make sure we can take 'len()' of it
			assert isinstance(config['agents'], list), f'The "agents" field must be of type list but was {type(config["agents"])}'
			assert len(config['agents']) == 1, f'Only one agent is permitted for this task, but {len(config["agents"])} were given.'

		self.check_types(config, config['task'])

	def _parse_and_set_agents(self, agent_list: list, needs_modelfile: bool) -> None:
		"""
		Utility function that gets the class of the agents and parses the provided arguments,
		making sure they are the correct type for each agent.

		Args:
			agent_list (list): The agents in the config for which to parse the arguments.
			needs_modelfile (bool): Whether or not RL-agents need modelfiles in this config.
		"""
		for agent in agent_list:
			# parse the provided string into the class
			agent['agent_class'] = get_class(agent['agent_class'])

			# This if-else contains the parsing logic for the different types of arguments agents can have, e.g. modelfiles or fixed-price-lists
			if needs_modelfile and issubclass(agent['agent_class'], (QLearningAgent, ActorCriticAgent)):
				assert isinstance(agent['argument'], str), \
					f'The "argument" field of this agent ({agent["name"]}) must be a string but was ({type(agent["argument"])})'
				assert agent['argument'].endswith('.dat'), \
					f'The "argument" field must contain a modelfile and therefore end in ".dat": {agent["argument"]}'
				# Check that the modelfile exists. Taken from am_configuration::_get_modelfile_path()
				full_path = os.path.abspath(os.path.join(PathManager.data_path, agent['argument']))
				assert os.path.exists(full_path), f'the specified modelfile does not exist: {full_path}'

			elif issubclass(agent['agent_class'], FixedPriceAgent):
				assert isinstance(agent['argument'], list), \
					f'The "argument" field of this agent ({agent["name"]}) must be a list but was ({type(agent["argument"])})'
				# Subclasses of FixedPriceAgent solely accept tuples
				agent['argument'] = tuple(agent['argument'])

			# check if some argument was provided even though an empty string should have been passed
			else:
				assert agent['argument'] == '', f'For agent "{agent["name"]}" no argument should have been passed, but got "{agent["argument"]}"!'

		self.agent = agent_list

	def _set_marketplace(self, marketplace_string: str) -> None:
		"""
		Utility function that validates the type of marketplace passed and sets the instance variable.

		Args:
			marketplace (str): The string of the class within the config dictionary.
		"""
		self.marketplace = get_class(marketplace_string)
		assert issubclass(self.marketplace, SimMarket), \
			f'The type of the passed marketplace must be a subclass of SimMarket: {self.marketplace}'

	def _assert_agent_marketplace_fit(self) -> None:
		"""
		Utility function that makes sure the agent(s) and marketplace are of the same type.
		"""

		assert all(issubclass(agent['agent_class'], CircularAgent) == issubclass(self.marketplace, CircularEconomy) for agent in self.agent), \
			f'The agents and marketplace must be of the same economy type (Linear/Circular): {self.agent} and {self.marketplace}'

	def _validate_config(self, config: dict, single_agent: bool, needs_modelfile: bool) -> None:
		"""
		Validate the given configuration dictionary and set the instance variables accordingly.

		Args:
			config (dict): The config dictionary to be validated.
			single_agent (bool): Whether or not only one agent should be used.
			needs_modelfile (bool): Whether or not the config must include modelfiles.

		Raises:
			AssertionError: In case the provided configuration is invalid.
		"""

		self._check_config_structure(config, single_agent)

		self._set_marketplace(config['marketplace'])

		self._parse_and_set_agents(config['agents'], needs_modelfile)

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

		# Since we only have one agent we extract it from the provided list
		# TODO: In #370 we can have more than one agent, since the rest are competitors
		self.agent = self.agent[0]
		assert issubclass(self.agent['agent_class'], (QLearningAgent, ActorCriticAgent)), \
			f'The agent must be a subclass of either QLearningAgent or ActorCriticAgent: {self.agent}'

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

		super(AgentMonitoringEnvironmentConfig, self)._validate_config(config, single_agent=False, needs_modelfile=True)

		self.enable_live_draw = config['enable_live_draw']
		self.episodes = config['episodes']
		self.plot_interval = config['plot_interval']

		# Since the agent_monitoring does not accept the dictionary but instead wants a list of tuples, we need to adapt the dictionary
		passed_agents = self.agent
		self.agent = []
		for current_agent in passed_agents:
			# with modelfile
			if issubclass(current_agent['agent_class'], (QLearningAgent, ActorCriticAgent)):
				self.agent.append((current_agent['agent_class'], [current_agent['argument'], current_agent['name']]))
			# without modelfile
			else:
				self.agent.append((current_agent['agent_class'], [current_agent['name']]))

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

	@classmethod
	def load(cls, filename: str) -> EnvironmentConfig:
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

	@classmethod
	def validate(cls, config: dict) -> EnvironmentConfig:
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


if __name__ == '__main__':  # pragma: no cover
	config: ExampleprinterEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_exampleprinter')
	print(config)
	print()
	config: AgentMonitoringEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_agent_monitoring')
	print(config)
	print()
	config: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	print(config)
