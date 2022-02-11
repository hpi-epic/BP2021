# helper
import json
import os
from abc import ABC, abstractmethod


class EnvironmentConfig(ABC):
	agents = None
	marketplace = None

	def load_config(filename='environment_config') -> dict:
		"""
		Load the configuration json file from the specified path.

		Args:
			filename (str, optional): The name of the json file containing the configuration values.
			Must be located in the BP2021/ folder. Defaults to 'environment_config'.

		Returns:
			dict: A dictionary containing the configuration values.
		"""
		filename += '.json'
		path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, filename)
		with open(path) as config_file:
			return json.load(config_file)

	@abstractmethod
	def get_task() -> str:
		"""
		Return the type of task this Config is for.

		Returns:
			str: The task name.
		"""
		raise NotImplementedError('This method os abstract. Use a subclass')


class TrainingEnvironmentConfig(EnvironmentConfig):

	def get_task() -> str:
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
