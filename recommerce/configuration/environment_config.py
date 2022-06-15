# helper
import json
import os
from abc import ABC, abstractmethod

import numpy as np

from recommerce.configuration.path_manager import PathManager
from recommerce.configuration.utils import get_class
from recommerce.market.circular.circular_sim_market import CircularEconomy
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_sim_market import LinearEconomy
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import FixedPriceAgent
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


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
				'separate_markets': False,
				'episodes': False,
				'plot_interval': False,
				'marketplace': False,
				'agents': False
			}
		else:
			raise AssertionError(f'The given level does not exist in an environment-config: {dict_key}')

	@classmethod
	def check_types(cls, config: dict, task: str = 'None', single_agent: bool = False, must_contain: bool = True) -> None:
		"""
		Check if all given variables have the correct types.
		If must_contain is True, all keys must exist, else non-existing keys will be skipped.

		Args:
			config (dict): The config to check.
			task (str): The task for which the variables should be checked.
			single_agent (bool): Whether or not only one agent is permitted.
			must_contain (bool, optional): Whether or not all variables must be present in the config. Defaults to True.

		Raises:
			AssertionError: If a key's value has an incorrect type or the marketplace or an agent could not be parsed to a valid class.
			KeyError: If the dictionary is missing a key but should contain all keys.
		"""
		if task in {'None', 'agent_monitoring'}:
			types_dict = {
				'task': str,
				'episodes': int,
				'plot_interval': int,
				'separate_markets': bool,
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
					if single_agent:
						assert len(config['agents']) == 1, f'Only one agent is permitted for this task, but {len(config["agents"])} were given.'
					for agent in config['agents']:
						# check types of the entries in the current agent dictionary
						for checked_key, checked_value in types_dict_agents.items():
							assert isinstance(agent[checked_key], checked_value), \
								f'{checked_key} must be a {checked_value} but was {type(agent[checked_key])}'
						try:
							get_class(agent['agent_class'])
						except Exception as error:
							raise AssertionError(f'This agent could not be parsed to a valid class: "{agent["agent_class"]}"') from error
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

		self.check_types(config, config['task'], single_agent)

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
			if needs_modelfile and issubclass(agent['agent_class'], ReinforcementLearningAgent):
				assert isinstance(agent['argument'], str), \
					f'The "argument" field of this agent ({agent["name"]}) must be a string but was ({type(agent["argument"])})'
				assert agent['argument'].endswith('.dat') or agent['argument'].endswith('.zip'), \
					f'The "argument" field must contain a modelfile and therefore end in ".dat" or ".zip": {agent["argument"]}'
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
		if issubclass(self.marketplace, CircularEconomy):
			assert all(issubclass(agent['agent_class'], CircularAgent) for agent in self.agent), \
				f'The marketplace ({self.marketplace}) is circular, so all agents need to be circular agents {self.agent}'
		elif issubclass(self.marketplace, LinearEconomy):
			assert all(issubclass(agent['agent_class'], LinearAgent) for agent in self.agent), \
				f'The marketplace ({self.marketplace}) is linear, so all agents need to be linear agents {self.agent}'

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
		super(TrainingEnvironmentConfig, self)._validate_config(config, single_agent=False, needs_modelfile=False)

		# Make sure the first given agent is a valid agent that can be trained
		assert issubclass(self.agent[0]['agent_class'], ReinforcementLearningAgent), \
			f'The first agent must be a ReinforcementLearningAgent: {self.agent}'

		# If we get more than one agent, make sure the rest are valid competitors and that we have the right amount
		if len(self.agent) > 1:
			assert self.marketplace.get_num_competitors() == np.inf or len(self.agent)-1 == self.marketplace.get_num_competitors(), \
				f'The number of competitors given is invalid: was {len(self.agent)-1} but should be {self.marketplace.get_num_competitors()}'

			for agent in self.agent[1:]:
				assert str(agent['agent_class'])[8:-2] in self.marketplace.get_competitor_classes(), \
					f'{agent["agent_class"]} is not a valid competitor on a {self.marketplace} market'

	def _get_task(self) -> str:
		return 'training'


class AgentMonitoringEnvironmentConfig(EnvironmentConfig):
	"""
	The environment configuration class for the agent_monitoring configuration.

	Instance variables:
		task (str): The task this config can be used for. Always "agent_monitoring".
		episodes (int): The number of episodes to run the monitoring for.
		plot_interval (int): The interval between plot creation.
		separate_markets (bool): If agents should be playing on separate marketplaces.
		marketplace (SimMarket subclass): A subclass of SimMarket, what marketplace the monitoring session should be run on.
		agent (list of tuples): A list containing the agents that should be trained.
			Each entry in the list is a tuple with the first item being the agent class, the second being a list.
			If the agent needs a modelfile, this will be the first entry in the list, the other entry is always an informal name for the agent.
	"""
	def _validate_config(self, config: dict) -> None:

		super(AgentMonitoringEnvironmentConfig, self)._validate_config(config, single_agent=False, needs_modelfile=True)

		self.episodes = config['episodes']
		self.plot_interval = config['plot_interval']
		self.separate_markets = config['separate_markets']
		self.competitors = None

		# If we get more than one agent and all agents play on the same market, make sure that we have the right amount
		if not self.separate_markets and len(self.agent) > 1:
			assert self.marketplace.get_num_competitors() == np.inf or len(self.agent)-1 == self.marketplace.get_num_competitors(), \
				f'The number of competitors given is invalid: was {len(self.agent)-1} but should be {self.marketplace.get_num_competitors()}'

		# It is possible to add custom competitors if separate_markets is True
		if self.separate_markets and 'competitors' in config:
			assert self.marketplace.get_num_competitors() == np.inf or len(config['competitors']) == self.marketplace.get_num_competitors(), \
				f'The number of competitors given is invalid: was {len(config["competitors"])} but should be {self.marketplace.get_num_competitors()}'
			self.competitors = [get_class(competitor) for competitor in config['competitors']]

		# Since the agent_monitoring does not accept the dictionary but instead wants a list of tuples, we need to adapt the dictionary
		passed_agents = self.agent
		self.agent = []
		for current_agent in passed_agents:
			# with modelfile
			if issubclass(current_agent['agent_class'], (ReinforcementLearningAgent, FixedPriceAgent)):
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
		super(ExampleprinterEnvironmentConfig, self)._validate_config(config, single_agent=False, needs_modelfile=True)

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
		Load the configuration json file from the `configuration_files` folder and instantiate the correct configuration class.

		Args:
			filename (str): The name of the json file containing the configuration values.
				Must be located in the `configuration_files` directory in the user's datapath folder.

		Returns:
			EnvironmentConfig: A subclass instance of EnvironmentConfig.
		"""
		filename += '.json'
		path = os.path.join(PathManager.user_path, 'configuration_files', filename)
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
	config_environment_exampleprinter: ExampleprinterEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_exampleprinter')
	print(config_environment_exampleprinter)
	print()
	config_environment_am: AgentMonitoringEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_agent_monitoring')
	print(config_environment_am)
	print()
	config_environment_training: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	print(config_environment_training)
