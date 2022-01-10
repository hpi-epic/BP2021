import random

import numpy as np

import configuration.config as config


def shuffle_quality() -> int:
	return min(max(int(np.random.normal(config.MAX_QUALITY / 2, 2 * config.MAX_QUALITY / 5)), 1), config.MAX_QUALITY)


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
	newdict = {}
	for key in dict1:
		if isinstance(dict1[key], dict):
			newdict[key] = divide_content_of_dict(dict1[key], divisor)
		else:
			assert isinstance(dict1[key], int) or isinstance(dict1[key], float), 'the dictionary should only contain numbers (int or float)'
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
	# TODO: assert dicts have the same structure
	newdict = {}
	for key in dict1:
		if isinstance(dict1[key], dict):
			newdict[key] = add_content_of_two_dicts(dict1[key], dict2[key])
		else:
			assert isinstance(dict1[key], int) or isinstance(dict1[key], float), 'dict1 should only contain numbers (int or float)'
			assert isinstance(dict2[key], int) or isinstance(dict2[key], float), 'dict2 should only contain numbers (int or float)'
			newdict[key] = dict1[key] + dict2[key]
	return newdict
