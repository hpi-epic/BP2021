#!/usr/bin/env python3

# helper
import json
import os

from alpha_business.configuration.path_manager import PathManager


class HyperparameterConfig():
	_instance = None

	def __new__(cls, config: dict = None):
		"""
		This function makes sure that the `HyperparameterConfig` is a singleton.

		Returns:
			HyperparameterConfig: The HyperparameterConfig instance.
		"""
		if cls._instance is None:
			# print('A new instance of HyperparameterConfig is being initialized')
			cls._instance = super(HyperparameterConfig, cls).__new__(cls)
			cls._instance._validate_and_set_config(config)
		else:
			print('An instance of HyperparameterConfig already exists and it will not be overwritten')

		return cls._instance

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
		self._set_rl_variables(config['rl'])
		self._set_sim_market_variables(config['sim_market'])

	def _check_config_rl_completeness(self, config: dict) -> None:
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

	def _check_config_sim_market_completeness(self, config: dict) -> None:
		"""
		Check if the passed config dictionary contains all sim_market values.

		Args:
			config (dict): The dictionary to be checked.
		"""
		assert 'max_storage' in config, 'your config is missing max_storage'
		assert 'episode_size' in config, 'your config is missing episode_size'
		assert 'max_price' in config, 'your config is missing max_price'
		assert 'max_quality' in config, 'your config is missing max_quality'
		assert 'number_of_customers' in config, 'your config is missing number_of_customers'
		assert 'production_price' in config, 'your config is missing production_price'
		assert 'storage_cost_per_product' in config, 'your config is missing storage_cost_per_product'

	def _set_rl_variables(self, config: dict) -> None:
		"""
		Update the global variables with new values provided by the config.

		Args:
			config (dict): The dictionary from which to read the new values.
		"""
		assert config['learning_rate'] > 0 and config['learning_rate'] < 1, 'learning_rate should be between 0 and 1 (excluded)'
		assert config['gamma'] >= 0 and config['gamma'] < 1, 'gamma should be between 0 (included) and 1 (excluded)'
		assert config['batch_size'] > 0, 'batch_size should be greater than 0'
		assert config['replay_size'] > 0, 'replay_size should be greater than 0'
		assert config['sync_target_frames'] > 0, 'sync_target_frames should be greater than 0'
		assert config['replay_start_size'] > 0, 'replay_start_size should be greater than 0'
		assert config['epsilon_decay_last_frame'] >= 0, 'epsilon_decay_last_frame should not be negative'

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
		assert config['max_storage'] >= 0, 'max_storage must be positive'
		assert config['number_of_customers'] > 0 and config['number_of_customers'] % 2 == 0, 'number_of_customers should be even and positive'
		assert config['production_price'] <= config['max_price'] and config['production_price'] >= 0, \
			'production_price needs to be smaller than max_price and >=0'
		assert config['max_quality'] > 0, 'max_quality should be positive'
		assert config['max_price'] > 0, 'max_price should be positive'
		assert config['episode_size'] > 0, 'episode_size should be positive'
		assert config['storage_cost_per_product'] >= 0, 'storage_cost_per_product should be non-negative'

		self.max_storage = config['max_storage']
		self.episode_length = config['episode_size']
		self.max_price = config['max_price']
		self.max_quality = config['max_quality']
		self.number_of_customers = config['number_of_customers']
		self.production_price = config['production_price']
		self.storage_cost_per_product = config['storage_cost_per_product']

		self.mean_reward_bound = self.episode_length * self.max_price * self.number_of_customers


class HyperparameterConfigLoader():

	def load(filename: str) -> HyperparameterConfig:
		"""
		Load the configuration json file from the specified path and instantiate a `HyperparameterConfig` object.

		Args:
			filename (str): The name of the json file containing the configuration values.
			Must be located in the BP2021/ folder.

		Returns:
			HyperparameterConfig: An instance of `HyperparameterConfig`.
		"""
		# in case there already is an instance of the config, we do not need to load the file again
		if HyperparameterConfig._instance is not None:
			return HyperparameterConfig()

		filename += '.json'
		path = os.path.join(PathManager.data_path, filename)
		with open(path) as config_file:
			config = json.load(config_file)
		return HyperparameterConfig(config)


config: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')

if __name__ == '__main__':  # pragma: no cover
	print(config)
