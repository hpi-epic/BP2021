#!/usr/bin/env python3

# helper
import json
import os

from recommerce.configuration.path_manager import PathManager


class HyperparameterConfig():

	def __init__(self, config):
		self._validate_and_set_config(config)

	@classmethod
	def get_required_fields(cls, dict_key) -> dict:
		"""
		Utility function that returns all of the keys required for a hyperparameter_config.json at the given level.
		The value of any given key indicates whether or not it is the key of a dictionary within the config (i.e. they are a level themselves).

		Args:
			dict_key (str): The key for which the required fields are needed. 'top-dict' for getting the keys of the first level.
				'top-dict', 'rl' or 'sim_market'.

		Returns:
			dict: The required keys for the config at the given level, together with a boolean indicating of they are the key
				of another level.

		Raises:
			AssertionError: If the given level is invalid.
		"""
		if dict_key == 'top-dict':
			return {'rl': True, 'sim_market': True}
		elif dict_key == 'rl':
			return {
				'gamma': False,
				'batch_size': False,
				'replay_size': False,
				'learning_rate': False,
				'sync_target_frames': False,
				'replay_start_size': False,
				'epsilon_decay_last_frame': False,
				'epsilon_start': False,
				'epsilon_final': False
			}
		elif dict_key == 'sim_market':
			return {
				'max_storage': False,
				'episode_length': False,
				'max_price': False,
				'max_quality': False,
				'number_of_customers': False,
				'production_price': False,
				'storage_cost_per_product': False
			}
		else:
			raise AssertionError(f'The given level does not exist in a hyperparameter-config: {dict_key}')

	def __str__(self) -> str:
		"""
		This overwrites the internal function that get called when you call `print(class_instance)`.

		Instead of printing the class name, prints the instance variables as a dictionary.

		Returns:
			str: The instance variables as a dictionary.
		"""
		return f'{self.__class__.__name__}: {self.__dict__}'

	def _validate_and_set_config(self, config: dict) -> None:
		"""
		Validate the given config dictionary and set the instance variables.

		Args:
			config (dict): The config to validate and take the values from.
		"""
		assert 'rl' in config, 'The config must contain an "rl" field.'
		assert 'sim_market' in config, 'The config must contain a "sim_market" field.'

		self._check_config_rl_completeness(config['rl'])
		self._check_config_sim_market_completeness(config['sim_market'])
		self.check_types(config, 'top-dict')
		self.check_types(config['rl'], 'rl')
		self.check_types(config['sim_market'], 'sim_market')
		self.check_rl_ranges(config['rl'])
		self.check_sim_market_ranges(config['sim_market'])
		self._set_rl_variables(config['rl'])
		self._set_sim_market_variables(config['sim_market'])

	@classmethod
	def _check_config_rl_completeness(cls, config: dict) -> None:
		"""
		Check if the passed config dictionary contains all rl values.

		Args:
			config (dict): The dictionary to be checked.
		"""
		assert 'gamma' in config, 'your config_rl is missing gamma'
		assert 'batch_size' in config, 'your config_rl is missing batch_size'
		assert 'replay_size' in config, 'your config_rl is missing replay_size'
		assert 'learning_rate' in config, 'your config_rl is missing learning_rate'
		assert 'sync_target_frames' in config, 'your config_rl is missing sync_target_frames'
		assert 'replay_start_size' in config, 'your config_rl is missing replay_start_size'
		assert 'epsilon_decay_last_frame' in config, 'your config_rl is missing epsilon_decay_last_frame'
		assert 'epsilon_start' in config, 'your config_rl is missing epsilon_start'
		assert 'epsilon_final' in config, 'your config_rl is missing epsilon_final'

	@classmethod
	def _check_config_sim_market_completeness(cls, config: dict) -> None:
		"""
		Check if the passed config dictionary contains all sim_market values.

		Args:
			config (dict): The dictionary to be checked.
		"""
		assert 'max_storage' in config, 'your config is missing max_storage'
		assert 'episode_length' in config, 'your config is missing episode_length'
		assert 'max_price' in config, 'your config is missing max_price'
		assert 'max_quality' in config, 'your config is missing max_quality'
		assert 'number_of_customers' in config, 'your config is missing number_of_customers'
		assert 'production_price' in config, 'your config is missing production_price'
		assert 'storage_cost_per_product' in config, 'your config is missing storage_cost_per_product'

	@classmethod
	def check_types(cls, config: dict, key: str, must_contain: bool = True) -> None:
		"""
		Check if all given variables have the correct types.
		If must_contain is True, all keys must exist, else non-existing keys will be skipped.

		Args:
			config (dict): The config to check.
			key (str): The key for which to check the values. 'top-dict', 'rl' or 'sim_market'.
			must_contain (bool, optional): Whether or not all variables must be present in the config. Defaults to True.

		Raises:
			KeyError: If the dictionary is missing a key but should contain all keys.
		"""
		if key == 'top-dict':
			types_dict = {
				'rl': dict,
				'sim_market': dict
			}
		elif key == 'rl':
			types_dict = {
				'gamma': (int, float),
				'batch_size': int,
				'replay_size': int,
				'learning_rate': (int, float),
				'sync_target_frames': int,
				'replay_start_size': int,
				'epsilon_decay_last_frame': int,
				'epsilon_start': (int, float),
				'epsilon_final': (int, float)
			}
		elif key == 'sim_market':
			types_dict = {
				'max_storage': int,
				'episode_length': int,
				'max_price': int,
				'max_quality': int,
				'number_of_customers': int,
				'production_price': int,
				'storage_cost_per_product': float
			}
		else:
			raise AssertionError(f'Your config contains an invalid key: {key}')

		for key, value in types_dict.items():
			try:
				assert isinstance(config[key], value), f'{key} must be a {value} but was {type(config[key])}'
			except KeyError as error:
				if must_contain:
					raise KeyError(f'Your config is missing the following required key: {key}') from error

	@classmethod
	def check_rl_ranges(cls, config: dict, must_contain: bool = True) -> None:
		"""
		Check if all rl variables are within their (pre-defined) ranges.

		Args:
			config (dict): The config for which to check the variables.
			must_contain (bool, optional): Whether or not all variables must be present in the config. Defaults to True.
		"""
		if must_contain or 'gamma' in config:
			assert config['gamma'] >= 0 and config['gamma'] < 1, 'gamma should be between 0 (included) and 1 (excluded)'
		if must_contain or 'batch_size' in config:
			assert config['batch_size'] > 0, 'batch_size should be greater than 0'
		if must_contain or 'replay_size' in config:
			assert config['replay_size'] > 0, 'replay_size should be greater than 0'
		if must_contain or 'learning_rate' in config:
			assert config['learning_rate'] > 0 and config['learning_rate'] < 1, 'learning_rate should be between 0 and 1 (excluded)'
		if must_contain or 'sync_target_frames' in config:
			assert config['sync_target_frames'] > 0, 'sync_target_frames should be greater than 0'
		if must_contain or 'replay_start_size' in config:
			assert config['replay_start_size'] > 0, 'replay_start_size should be greater than 0'
		if must_contain or 'epsilon_decay_last_frame' in config:
			assert config['epsilon_decay_last_frame'] >= 0, 'epsilon_decay_last_frame should not be negative'
		if must_contain or 'epsilon_start' in config:
			assert config['epsilon_start'] > 0 and config['epsilon_start'] <= 1, 'epsilon_start should be between 0 and 1 (excluded)'
		if must_contain or 'epsilon_final' in config:
			assert config['epsilon_final'] > 0 and config['epsilon_final'] <= 1, 'epsilon_final should be between 0 and 1 (excluded)'
		if must_contain or ('epsilon_start' in config and 'epsilon_final' in config):
			assert config['epsilon_start'] > config['epsilon_final'], 'epsilon_start should be greater than epsilon_final'

	@classmethod
	def check_sim_market_ranges(cls, config: dict, must_contain: bool = True) -> None:
		"""
		Check if all sim_market variables are within their (pre-defined) ranges.

		Args:
			config (dict): The config for which to check the variables.
			must_contain (bool, optional): Whether or not all variables must be present in the config. Defaults to True.
		"""
		if must_contain or 'max_storage' in config:
			assert config['max_storage'] >= 0, 'max_storage must be positive'
		if must_contain or 'number_of_customers' in config:
			assert config['number_of_customers'] > 0 and config['number_of_customers'] % 2 == 0, 'number_of_customers should be even and positive'
		if must_contain or 'production_price' in config:
			assert config['production_price'] <= config['max_price'] and config['production_price'] >= 0, \
				'production_price needs to be smaller than max_price and >=0'
		if must_contain or 'max_quality' in config:
			assert config['max_quality'] > 0, 'max_quality should be positive'
		if must_contain or 'max_price' in config:
			assert config['max_price'] > 0, 'max_price should be positive'
		if must_contain or 'episode_length' in config:
			assert config['episode_length'] > 0, 'episode_length should be positive'
		if must_contain or 'storage_cost_per_product' in config:
			assert config['storage_cost_per_product'] >= 0, 'storage_cost_per_product should be non-negative'

	def _set_rl_variables(self, config: dict) -> None:
		"""
		Update the global variables with new values provided by the config.

		Args:
			config (dict): The dictionary from which to read the new values.
		"""
		self.gamma = config['gamma']
		self.learning_rate = config['learning_rate']
		self.batch_size = config['batch_size']
		self.replay_size = config['replay_size']
		self.sync_target_frames = config['sync_target_frames']
		self.replay_start_size = config['replay_start_size']

		self.epsilon_decay_last_frame = config['epsilon_decay_last_frame']
		self.epsilon_start = config['epsilon_start']
		self.epsilon_final = config['epsilon_final']

	def _set_sim_market_variables(self, config: dict) -> None:
		"""
		Update the global variables with new values provided by the config.

		Args:
			config (dict): The dictionary from which to read the new values.
		"""

		self.max_storage = config['max_storage']
		self.episode_length = config['episode_length']
		self.max_price = config['max_price']
		self.max_quality = config['max_quality']
		self.number_of_customers = config['number_of_customers']
		self.production_price = config['production_price']
		self.storage_cost_per_product = config['storage_cost_per_product']

		self.mean_return_bound = self.episode_length * self.max_price * self.number_of_customers


class HyperparameterConfigLoader():

	@classmethod
	def load(cls, filename: str) -> HyperparameterConfig:
		"""
		Load the configuration json file from the specified path and instantiate a `HyperparameterConfig` object.

		Args:
			filename (str): The name of the json file containing the configuration values.
			Must be located in the user's datapath folder.

		Returns:
			HyperparameterConfig: An instance of `HyperparameterConfig`.
		"""
		filename += '.json'
		path = os.path.join(PathManager.user_path, filename)
		with open(path) as config_file:
			config = json.load(config_file)
		return HyperparameterConfig(config)
