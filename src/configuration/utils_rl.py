#!/usr/bin/env python3

# helper
import json
import os

GAMMA = None
BATCH_SIZE = None
REPLAY_SIZE = None
LEARNING_RATE = None
SYNC_TARGET_FRAMES = None
REPLAY_START_SIZE = None

EPSILON_DECAY_LAST_FRAME = None
EPSILON_START = None
EPSILON_FINAL = None


def load_config(path_rl=os.path.dirname(__file__) + os.sep + '../..' + os.sep + 'config_rl.json') -> dict:
	"""
	Load the Reinforcement Learning json file from the specified path.

	Args:
		path_rl (str, optional): The path to the json file containing the configuration values. Defaults to os.path.dirname(__file__)+os.sep+'..'+os.sep+'config_rl.json'.

	Returns:
		dict: A dictionary containing the configuration values.
	"""
	with open(path_rl) as config_file:
		return json.load(config_file)


def check_config_completeness(config: dict) -> None:
	"""
	Check if the passed config dictionary contains all values.

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


def update_rl_variables(config: dict) -> None:
	"""
	Update the global variables with new values provided by the config.

	Args:
		config (dict): The dictionary from which to read the new values.
	"""
	listOfGlobals = globals()
	listOfGlobals['GAMMA'] = float(config['gamma'])
	listOfGlobals['LEARNING_RATE'] = float(config['learning_rate'])
	listOfGlobals['BATCH_SIZE'] = int(config['batch_size'])
	listOfGlobals['REPLAY_SIZE'] = int(config['replay_size'])
	listOfGlobals['SYNC_TARGET_FRAMES'] = int(config['sync_target_frames'])
	listOfGlobals['REPLAY_START_SIZE'] = int(config['replay_start_size'])

	listOfGlobals['EPSILON_DECAY_LAST_FRAME'] = int(config['epsilon_decay_last_frame'])
	listOfGlobals['EPSILON_START'] = float(config['epsilon_start'])
	listOfGlobals['EPSILON_FINAL'] = float(config['epsilon_final'])

	assert listOfGlobals['LEARNING_RATE'] > 0 and listOfGlobals['LEARNING_RATE'] < 1, 'learning_rate should be between 0 and 1 (excluded)'
	assert listOfGlobals['GAMMA'] >= 0 and listOfGlobals['GAMMA'] < 1, 'gamma should be between 0 (included) and 1 (excluded)'
	assert listOfGlobals['BATCH_SIZE'] > 0, 'batch_size should be greater than 0'
	assert listOfGlobals['REPLAY_SIZE'] > 0, 'replay_size should be greater than 0'
	assert listOfGlobals['SYNC_TARGET_FRAMES'] > 0, 'sync_target_frames should be greater than 0'
	assert listOfGlobals['REPLAY_START_SIZE'] > 0, 'replay_start_size should be greater than 0'
	assert listOfGlobals['EPSILON_DECAY_LAST_FRAME'] >= 0, 'epsilon_decay_last_frame should not be negative'


config = load_config()
check_config_completeness(config)
update_rl_variables(config)
