#!/usr/bin/env python3

# helper
import json
import os

# rl
GAMMA = None
BATCH_SIZE = None
REPLAY_SIZE = None
LEARNING_RATE = None
SYNC_TARGET_FRAMES = None
REPLAY_START_SIZE = None
EPSILON_DECAY_LAST_FRAME = None
EPSILON_START = None
EPSILON_FINAL = None

# sim_market
MAX_STORAGE = 100
STORAGE_COST_PER_PRODUCT = None
MAX_PRICE = None
MAX_QUALITY = None
MEAN_REWARD_BOUND = None
NUMBER_OF_CUSTOMERS = None
PRODUCTION_PRICE = None
EPISODE_LENGTH = None


class HyperparameterConfig():

	def __init__(self, config: dict):
		self._validate_config(config)

	def __str__(self) -> str:
		"""
		This overwrites the internal function that get called when you call `print(class_instance)`.

		Instead of printing the class name, prints the instance variables as a dictionary.

		Returns:
			str: The instance variables as a dictionary.
		"""
		return f'{self.__class__.__name__}: {self.__dict__}'

	def _validate_config(self, config: dict):
		assert 'rl' in config, 'The config must contain an "rl" field.'
		assert 'sim_market' in config, 'The config must contain a "sim_market" field.'

		self._check_config_rl_completeness(config['rl'])
		self._check_config_sim_market_completeness(config['sim_market'])
		self._update_rl_variables(config['rl'])
		self._update_sim_market_variables(config['sim_market'])

	def _check_config_rl_completeness(self, config: dict) -> None:
		"""
		Check if the passed config dictionary contains all rl values.

		Args:
			config (dict): The dictionary to be checked.
		"""
		# ordered like in the config_rl.json
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
		assert 'episode_size' in config, 'your config is missing episode_size'
		assert 'max_price' in config, 'your config is missing max_price'
		assert 'max_quality' in config, 'your config is missing max_quality'
		assert 'number_of_customers' in config, 'your config is missing number_of_customers'
		assert 'production_price' in config, 'your config is missing production_price'
		assert 'storage_cost_per_product' in config, 'your config is missing storage_cost_per_product'

	def _update_rl_variables(self, config: dict) -> None:
		"""
		Update the global variables with new values provided by the config.

		Args:
			config (dict): The dictionary from which to read the new values.
		"""
		self.gamma = float(config['gamma'])
		self.learning_rate = float(config['learning_rate'])
		self.batch_size = int(config['batch_size'])
		self.replay_size = int(config['replay_size'])
		self.sync_target_frames = int(config['sync_target_frames'])
		self.replay_start_size = int(config['replay_start_size'])

		self.epsilon_decay_last_frame = int(config['epsilon_decay_last_frame'])
		self.epsilon_start = float(config['epsilon_start'])
		self.epsilon_final = float(config['epsilon_final'])

		assert self.learning_rate > 0 and self.learning_rate < 1, \
			'learning_rate should be between 0 and 1 (excluded)'
		assert self.gamma >= 0 and self.gamma < 1, 'gamma should be between 0 (included) and 1 (excluded)'
		assert self.batch_size > 0, 'batch_size should be greater than 0'
		assert self.replay_size > 0, 'replay_size should be greater than 0'
		assert self.sync_target_frames > 0, 'sync_target_frames should be greater than 0'
		assert self.replay_start_size > 0, 'replay_start_size should be greater than 0'
		assert self.epsilon_decay_last_frame >= 0, 'epsilon_decay_last_frame should not be negative'

	def _update_sim_market_variables(self, config: dict) -> None:
		"""
		Update the global variables with new values provided by the config.

		Args:
			config (dict): The dictionary from which to read the new values.
		"""
		self.episode_length = int(config['episode_size'])

		self.max_price = int(config['max_price'])
		self.max_quality = int(config['max_quality'])
		self.number_of_customers = int(config['number_of_customers'])
		self.production_price = int(config['production_price'])
		self.storage_cost_per_product = config['storage_cost_per_product']

		assert self.number_of_customers > 0 and self.number_of_customers % 2 == 0, 'number_of_customers should be even and positive'
		assert self.production_price <= self.max_price and self.production_price >= 0, 'production_price needs to smaller than max_price and >= 0'
		assert self.max_quality > 0, 'max_quality should be positive'
		assert self.max_price > 0, 'max_price should be positive'
		assert self.episode_length > 0, 'episode_size should be positive'
		assert self.storage_cost_per_product >= 0, 'storage_cost_per_product should be non-negative'

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
		filename += '.json'
		path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, filename)
		with open(path) as config_file:
			config = json.load(config_file)
		return HyperparameterConfig(config)


if __name__ == '__main__':  # pragma: no cover
	config: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')
	print(config)
