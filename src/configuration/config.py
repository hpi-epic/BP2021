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
STORAGE_COST_PER_PRODUCT = 0.5
MAX_PRICE = None
MAX_QUALITY = None
MEAN_REWARD_BOUND = None
NUMBER_OF_CUSTOMERS = None
PRODUCTION_PRICE = None
EPISODE_LENGTH = None


def load_config(filename='config') -> dict:
	"""
	Load the configuration json file from the specified path.

	Args:
		path (str, optional): The path to the json file containing the configuration values. Defaults to os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'config.json').

	Returns:
		dict: A dictionary containing the configuration values.
	"""
	filename += '.json'
	path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, filename)
	with open(path) as config_file:
		return json.load(config_file)


def check_config_rl_completeness(config: dict) -> None:
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


def check_config_sim_market_completeness(config: dict) -> None:
	"""
	Check if the passed config dictionary contains all sim_market values.

	Args:
		config (dict): The dictionary to be checked.
	"""
	# ordered alphabetically in the config_sim_market.json
	assert 'episode_size' in config, 'your config is missing episode_size'
	assert 'max_price' in config, 'your config is missing max_price'
	assert 'max_quality' in config, 'your config is missing max_quality'
	assert 'number_of_customers' in config, 'your config is missing number_of_customers'
	assert 'production_price' in config, 'your config is missing production_price'


def update_rl_variables(config: dict) -> None:
	"""
	Update the global variables with new values provided by the config.

	Args:
		config (dict): The dictionary from which to read the new values.
	"""
	global_variables = globals()
	global_variables['GAMMA'] = float(config['gamma'])
	global_variables['LEARNING_RATE'] = float(config['learning_rate'])
	global_variables['BATCH_SIZE'] = int(config['batch_size'])
	global_variables['REPLAY_SIZE'] = int(config['replay_size'])
	global_variables['SYNC_TARGET_FRAMES'] = int(config['sync_target_frames'])
	global_variables['REPLAY_START_SIZE'] = int(config['replay_start_size'])

	global_variables['EPSILON_DECAY_LAST_FRAME'] = int(config['epsilon_decay_last_frame'])
	global_variables['EPSILON_START'] = float(config['epsilon_start'])
	global_variables['EPSILON_FINAL'] = float(config['epsilon_final'])

	assert global_variables['LEARNING_RATE'] > 0 and global_variables['LEARNING_RATE'] < 1, 'learning_rate should be between 0 and 1 (excluded)'
	assert global_variables['GAMMA'] >= 0 and global_variables['GAMMA'] < 1, 'gamma should be between 0 (included) and 1 (excluded)'
	assert global_variables['BATCH_SIZE'] > 0, 'batch_size should be greater than 0'
	assert global_variables['REPLAY_SIZE'] > 0, 'replay_size should be greater than 0'
	assert global_variables['SYNC_TARGET_FRAMES'] > 0, 'sync_target_frames should be greater than 0'
	assert global_variables['REPLAY_START_SIZE'] > 0, 'replay_start_size should be greater than 0'
	assert global_variables['EPSILON_DECAY_LAST_FRAME'] >= 0, 'epsilon_decay_last_frame should not be negative'


def update_sim_market_variables(config: dict) -> None:
	"""
	Update the global variables with new values provided by the config.

	Args:
		config (dict): The dictionary from which to read the new values.
	"""
	global_variables = globals()
	global_variables['EPISODE_LENGTH'] = int(config['episode_size'])

	global_variables['MAX_PRICE'] = int(config['max_price'])
	global_variables['MAX_QUALITY'] = int(config['max_quality'])
	global_variables['NUMBER_OF_CUSTOMERS'] = int(config['number_of_customers'])
	global_variables['PRODUCTION_PRICE'] = int(config['production_price'])

	assert global_variables['NUMBER_OF_CUSTOMERS'] > 0 and global_variables['NUMBER_OF_CUSTOMERS'] % 2 == 0, 'number_of_customers should be even and positive'
	assert global_variables['PRODUCTION_PRICE'] <= global_variables['MAX_PRICE'] and global_variables['PRODUCTION_PRICE'] >= 0, 'production_price needs to smaller than max_price and positive or zero'
	assert global_variables['MAX_QUALITY'] > 0, 'max_quality should be positive'
	assert global_variables['MAX_PRICE'] > 0, 'max_price should be positive'
	assert global_variables['EPISODE_LENGTH'] > 0, 'episode_size should be positive'

	global_variables['MEAN_REWARD_BOUND'] = global_variables['EPISODE_LENGTH'] * global_variables['MAX_PRICE'] * global_variables['NUMBER_OF_CUSTOMERS']


config = load_config()
check_config_rl_completeness(config['rl'])
check_config_sim_market_completeness(config['sim_market'])
update_rl_variables(config['rl'])
update_sim_market_variables(config['sim_market'])
