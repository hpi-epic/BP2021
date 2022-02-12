# helper
import importlib
import json
import os
from abc import ABC, abstractmethod


class EnvironmentConfig(ABC):

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

	@abstractmethod
	def validate_config(config: dict) -> None:
		"""
		Validate the given configuration dictionary and set the instance variables accordingly.

		Args:
			config (dict): The config dictionary to be validated.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def get_task() -> str:
		"""
		Return the type of task this Config is for.

		Returns:
			str: The task name.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

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


class TrainingEnvironmentConfig(EnvironmentConfig):

	def validate_config(self, config: dict) -> None:
		assert 'agents' in config, f'The config must have an "agents" field: {config}'
		assert 'marketplace' in config, f'The config must have a "marketplace" field: {config}'

		assert isinstance(config['agents'], dict), \
			f'The "agents" field must be a dict: {config["agents"]} ({type(config["agents"])})'
		assert all(isinstance(config['agents'][agent], dict) for agent in config['agents']), \
			f'All agents in the "agents" field must be dictionaries: {[config["agents"][agent] for agent in config["agents"]]}'

		assert isinstance(config['marketplace'], str), \
			f'The "marketplace" field must be a str: {config["marketplace"]} ({type(config["marketplace"])})'

		self.marketplace = self.get_class(config['marketplace'])
		self.agents = [self.get_class(config['agents'][agent]['class']) for agent in config['agents']]

	def get_task(self) -> str:
		return 'training'


class AgentMonitoringEnvironmentConfig(EnvironmentConfig):
	enable_live_draw = None
	episodes = None
	plot_interval = None
	folder_path = None

	def get_task() -> str:
		return 'agent_monitoring'


class ExampleprinterEnvironmentConfig(EnvironmentConfig):

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
			EnvironmentConfig: A subclass isntance of EnvironmentConfig.
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
