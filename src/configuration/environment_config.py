# helper
import importlib
import json
import os
from abc import ABC, abstractmethod


class EnvironmentConfig(ABC):
	"""
	An abstract environment configuration class.
	"""

	def __init__(self, config: dict):
		self.validate_config(config)

	def __str__(self) -> str:
		"""
		This overwrites the internal function that get called when you call `print(class_instance)`.

		Instead of printing the class name, prints the instance variables as a dictionary.

		Returns:
			str: The instance variables as a dictionary.
		"""
		return f'{self.__class__.__name__}: {self.__dict__}'

	def validate_config(self, config: dict, single_agent: bool, needs_modelfile: bool) -> None:
		"""
		Validate the given configuration dictionary and set the instance variables accordingly.

		Args:
			config (dict): The config dictionary to be validated.
			single_agent (bool): Whether or not only one agent is needed.
			needs_modelfile (bool): Whether or not the config must include modelfiles.
		"""

		# CHECK: All required top-level fields exist
		assert 'agents' in config, f'The config must have an "agents" field: {config}'
		assert 'marketplace' in config, f'The config must have a "marketplace" field: {config}'

		# TODO: Assert marketplace and agent match

		# CHECK: Agents
		assert isinstance(config['agents'], dict), \
			f'The "agents" field must be a dict: {config["agents"]} ({type(config["agents"])})'

		# Shorten the agent dictionary if only one is necessary
		if single_agent and len(config['agents']) > 1:
			used_agent = list(config['agents'].items())[0]
			config['agents'] = {used_agent[0]: used_agent[1]}
			print(f'Multiple agents were provided but only the first one will be used:\n{config["agents"]}')

		# Save the agents in config['agents'] in a list for easier access
		agent_dictionaries = [config['agents'][agent] for agent in config['agents']]

		assert (isinstance(agent, dict) for agent in agent_dictionaries), \
			f'All agents in the "agents" field must be dictionaries: {[config["agents"][agent] for agent in config["agents"]]}'

		# CHECK: Agents::Class
		assert all('class' in agent for agent in agent_dictionaries), f'Each agent must have a "class" field: {agent_dictionaries}'
		assert all(isinstance(agent['class'], str) for agent in agent_dictionaries), \
			f'The "class" fields must be strings: {agent_dictionaries} ({[type(agent["class"]) for agent in agent_dictionaries]})'

		# CHECK: Agents::Modelfile
		if needs_modelfile:
			# TODO: Only check modelfiles for non-RL agents (need to check class inheritance, need to get the agent class beforehand)
			assert all('modelfile' in agent for agent in agent_dictionaries), f'Each agent must have a "modelfile" field: {agent_dictionaries}'
			# TODO: Verify if the modelfile is a .dat and in the data folder!
			assert all(isinstance(agent['modelfile'], str) for agent in agent_dictionaries), \
				f'The "modelfile" fields must be strings: {agent_dictionaries} ({[type(agent["modelfile"]) for agent in agent_dictionaries]})'

		# CHECK: Marketplace
		assert isinstance(config['marketplace'], str), \
			f'The "marketplace" field must be a str: {config["marketplace"]} ({type(config["marketplace"])})'

		self.marketplace = self.get_class(config['marketplace'])

		# FINAL: Assign instance variables
		# If a modelfile is needed, the self.agents will be a list of tuples (as required by agent_monitoring), else just a list of classes
		if needs_modelfile:
			self.agent = [(self.get_class(agent['class']), agent['modelfile']) for agent in agent_dictionaries]
		else:
			self.agent = [self.get_class(agent['class']) for agent in agent_dictionaries]
		# If only one agent is needed, we just use the first agent from the list we created before
		if single_agent:
			self.agent = self.agent[0]

	def get_class(self, import_string: str):
		"""
		Get the class from the given string.

		Args:
			import_string (str): A string containing the import path in the format 'module.submodule.class'.

		Returns:
			class: The imported class
		"""
		module_name, class_name = import_string.rsplit('.', 1)
		return getattr(importlib.import_module(module_name), class_name)

	@abstractmethod
	def get_task() -> str:
		"""
		Return the type of task this Config is for.

		Returns:
			str: The task name.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')


class TrainingEnvironmentConfig(EnvironmentConfig):
	"""
	The environment configuration class for the training configuration.
	"""

	def validate_config(self, config: dict) -> None:
		super(TrainingEnvironmentConfig, self).validate_config(config, single_agent=True, needs_modelfile=False)

	def get_task(self) -> str:
		return 'training'


class AgentMonitoringEnvironmentConfig(EnvironmentConfig):
	"""
	The environment configuration class for the agent_monitoring configuration.
	"""

	def validate_config(self, config: dict) -> None:

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

		# FINAL: Assign instance variables
		self.enable_live_draw = config['enable_live_draw']
		self.episodes = config['episodes']
		self.plot_interval = config['plot_interval']

		# We do the super call last because getting the classes takes longer than the other operations, so we save time in case of an error.
		super(AgentMonitoringEnvironmentConfig, self).validate_config(config, single_agent=False, needs_modelfile=True)

	def get_task() -> str:
		return 'agent_monitoring'


class ExampleprinterEnvironmentConfig(EnvironmentConfig):
	"""
	The environment configuration class for the exampleprinter configuration.
	"""

	def validate_config(self, config: dict) -> None:
		super(ExampleprinterEnvironmentConfig, self).validate_config(config, single_agent=True, needs_modelfile=True)

	def get_task() -> str:
		return 'exampleprinter'


class ConfigLoader():

	def load(self, filename: str = 'environment_config') -> EnvironmentConfig:
		"""
		Load the configuration json file from the specified path and instantiate the correct configuration class.

		Args:
			filename (str, optional): The name of the json file containing the configuration values.
			Must be located in the BP2021/ folder. Defaults to 'environment_config'.

		Returns:
			EnvironmentConfig: A subclass instance of EnvironmentConfig.
		"""
		filename += '.json'
		path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, filename)
		with open(path) as config_file:
			config = json.load(config_file)
		if config['task'] == 'training':
			return TrainingEnvironmentConfig(config)
		elif config['task'] == 'agent_monitoring':
			return AgentMonitoringEnvironmentConfig(config)
		elif config['task'] == 'exampleprinter':
			return ExampleprinterEnvironmentConfig(config)
		else:
			raise RuntimeError(f'The specified task is unknown: {config["task"]}\nConfig: {config}')


if __name__ == '__main__':
	config: EnvironmentConfig = ConfigLoader().load()
	print(config)
