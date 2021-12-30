#!/usr/bin/env python3

# helper
import json
import os
import random

import numpy as np

MAX_STORAGE = 100
STORAGE_COST_PER_PRODUCT = 0.5
MAX_PRICE = None
MAX_QUALITY = None
MEAN_REWARD_BOUND = None
NUMBER_OF_CUSTOMERS = None
PRODUCTION_PRICE = None
EPISODE_LENGTH = None

config = {}


def load_config(path_sim_market=os.path.dirname(__file__) + os.sep + '../..' + os.sep + 'config_sim_market.json'):
	"""
	Load the SimMarket json file from the specified path.

	Args:
		path_sim_market (str, optional): The path to the json file containing the configuration values. Defaults to os.path.dirname(__file__)+os.sep+'..'+os.sep+'config_sim_market.json'.

	Returns:
		dict: A dictionary containing the configuration values.
	"""
	with open(path_sim_market) as config_file:
		return json.load(config_file)


config = load_config()

# ordered alphabetically in the config_sim_market.json
assert 'episode_size' in config, 'your config is missing episode_size'
assert 'max_price' in config, 'your config is missing max_price'
assert 'max_quality' in config, 'your config is missing max_quality'
assert 'number_of_customers' in config, 'your config is missing number_of_customers'
assert 'production_price' in config, 'your config is missing production_price'

EPISODE_LENGTH = int(config['episode_size'])

MAX_PRICE = int(config['max_price'])
MAX_QUALITY = int(config['max_quality'])
NUMBER_OF_CUSTOMERS = int(config['number_of_customers'])
PRODUCTION_PRICE = int(config['production_price'])


assert NUMBER_OF_CUSTOMERS > 0 and NUMBER_OF_CUSTOMERS % 2 == 0, 'number_of_customers should be even and positive'
assert PRODUCTION_PRICE <= MAX_PRICE and PRODUCTION_PRICE >= 0, 'production_price needs to smaller than max_price and positive or zero'
assert MAX_QUALITY > 0, 'max_quality should be positive'
assert MAX_PRICE > 0, 'max_price should be positive'
assert EPISODE_LENGTH > 0, 'episode_size should be positive'

MEAN_REWARD_BOUND = EPISODE_LENGTH * MAX_PRICE * NUMBER_OF_CUSTOMERS


def shuffle_quality() -> int:
	return min(max(int(np.random.normal(MAX_QUALITY / 2, 2 * MAX_QUALITY / 5)), 1), MAX_QUALITY)


# The following methods should be library calls in the future.
def softmax(preferences) -> np.array:
	preferences = np.minimum(np.ones(len(preferences)) * 20, preferences)  # This avoids an overflow error in the next line
	exp_preferences = np.exp(preferences)
	return exp_preferences / sum(exp_preferences)


def shuffle_from_probabilities(probabilities) -> int:
	randomnumber = random.random()
	sum = 0
	for i, p in enumerate(probabilities):
		sum += p
		if randomnumber <= sum:
			return i
	return len(probabilities) - 1  # pragma: no cover


def write_dict_to_tensorboard(writer, dictionary, counter, is_cumulative=False) -> None:
	"""This function takes a dictionary of data with data from one step and adds it at the specified time to the tensorboard.

	Args:
		writer (SummaryWriter): The tensorboard writer on which the calls to write are taken.
		dictionary (dict): The dictionary containing the data to be added to the tensorboard.
		counter (int): Specifies the timestamp/step at which the data should be added to the tensorboard.
		is_cumulative (bool, optional): . Defaults to False.
	"""
	for name, content in dictionary.items():

		if is_cumulative:
			# do not print cumulative actions or states because it has no meaning
			if (name.startswith('actions') or name.startswith('state')):
				continue
			else:
				name = 'cumulated_' + name
		if isinstance(content, dict):
			writer.add_scalars(name, content, counter)
		else:
			writer.add_scalar(name, content, counter)


def divide_content_of_dict(dict1, divisor) -> dict:
	"""Recursively divide a dictionary which contains only numbers by a divisor

	Args:
		dict1 (dict): the dictionary you would like to divide
		divisor (number): the divisor

	Returns:
		dict: the dictionary containing the divided numbers
	"""
	# TODO: assert dictionary contains only numbers
	newdict = {}
	for key in dict1:
		if isinstance(dict1[key], dict):
			newdict[key] = divide_content_of_dict(dict1[key], divisor)
		else:
			newdict[key] = dict1[key] / divisor
	return newdict


def add_content_of_two_dicts(dict1, dict2) -> dict:
	"""This function takes two dicts and runs recursively through the dicts. The dicts must have the same structure.

	Args:
		dict1 (dict): first dictionary you want to add
		dict2 (dict): second dictionary you want to add

	Returns:
		dict: same structure as dict1 and dict2, each entry contains the sum of the entries of dict1 and dict2
	"""
	# TODO: assert dicts have the same structure, dictionary contains only numbers
	newdict = {}
	for key in dict1:
		if isinstance(dict1[key], dict):
			newdict[key] = add_content_of_two_dicts(dict1[key], dict2[key])
		else:
			newdict[key] = dict1[key] + dict2[key]
	return newdict


def write_content_of_dict_to_overview_svg(manipulator, dictionary, episode):
	for key in dictionary:
		print(key, dictionary[key])
	manipulator.replace_one_value('simulation_current_episode', str(episode))
	# manipulator.save_overview_svg(filename='test'+str(episode)+'.svg')
