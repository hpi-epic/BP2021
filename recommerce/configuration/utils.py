import importlib
import os
import random
import re

import numpy as np
from attrdict import AttrDict

from recommerce.configuration.path_manager import PathManager
from recommerce.monitoring.svg_manipulation import SVGManipulator


def ensure_results_folders_exist():
	"""
	Create the results directory as well as the needed subdirectories.

	If your code assumes that the results folder or any of its subfolders exist, call this function beforehand.
	"""
	folders = ['monitoring', 'exampleprinter', 'runs', 'trainedModels', 'policyanalyzer']
	for folder in folders:
		os.makedirs(os.path.join(PathManager.results_path, folder), exist_ok=True)


def shuffle_quality(config: AttrDict) -> int:
	return min(max(int(np.random.normal(config.max_quality / 2, 2 * config.max_quality / 5)), 1), config.max_quality)


def softmax(preferences: np.array) -> np.array:
	preferences = np.minimum(np.ones(len(preferences)) * 20, preferences)  # This avoids an overflow error in the next line
	exp_preferences = np.exp(preferences)
	return exp_preferences / sum(exp_preferences)


def shuffle_from_probabilities(probabilities: np.array) -> int:
	randomnumber = random.random()
	probability_sum = 0
	for i, p in enumerate(probabilities):
		probability_sum += p
		if randomnumber <= probability_sum:
			return i
	return len(probabilities) - 1


def cartesian_product(list_a, list_b):
	"""
	This helper function takes to lists and generates the cartesian product

	Args:
		list_a (list): The first list of objects
		list_b (list): The second list of objects

	Returns:
		list: List of tuples containing all combinations of list_a entries and list_b entries
	"""
	assert isinstance(list_a, list) and isinstance(list_b, list), 'You must give to lists'
	output_list = []
	for a in list_a:
		output_list.extend((a, b) for b in list_b)
	return output_list


def write_dict_to_tensorboard(writer, dictionary: dict, counter: int, is_cumulative: bool = False, episode_length: int = None) -> None:
	"""
	This function takes a dictionary of data with data from one step and adds it at the specified time to the tensorboard.

	Args:
		writer (SummaryWriter): The tensorboard writer on which the calls to write are taken.
		dictionary (dict): The dictionary containing the data to be added to the tensorboard.
		counter (int): Specifies the timestamp/step at which the data should be added to the tensorboard.
		is_cumulative (bool, optional): . Defaults to False.
	"""
	assert is_cumulative == (episode_length is not None), 'Episode length must be exactly specified if is_cumulative is True'
	for name, content in dictionary.items():
		if is_cumulative:
			# do not print cumulative actions or states because it has no meaning
			if (name.startswith('actions') or name.startswith('state')):
				name = f'average_{name}'
				if isinstance(content, dict):
					content = divide_content_of_dict(content, episode_length)
				else:
					content = content / episode_length
			else:
				name = f'cumulated_{name}'
		if isinstance(content, dict):
			writer.add_scalars(name, content, counter)
		else:
			writer.add_scalar(name, content, counter)


def divide_content_of_dict(dict1: dict, divisor) -> dict:
	"""
	Recursively divide a dictionary which contains only numbers by a divisor

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
			assert isinstance(dict1[key], (int, float, np.float32)), f'the dictionary should only contain numbers (int or float): {dict1}'
			newdict[key] = dict1[key] / divisor
	return newdict


def add_content_of_two_dicts(dict1, dict2) -> dict:
	"""
	This function takes two dicts and runs recursively through the dicts. The dicts must have the same structure.

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
			assert isinstance(dict1[key], (int, float, np.float32)), f'dict1 should only contain numbers (int or float): {dict1}'
			assert isinstance(dict2[key], (int, float, np.float32)), f'dict2 should only contain numbers (int or float): {dict2}'
			newdict[key] = dict1[key] + dict2[key]
	return newdict


def convert_dict_to_float(dict1: dict) -> dict:
	"""
	This function takes a dict and recursively converts all entries to float.

	Args:
		dict1 (dict): the dict you want to convert

	Returns:
		dict: same structure as dict1, but all entries are floats
	"""
	newdict = {}
	for key in dict1:
		if isinstance(dict1[key], dict):
			newdict[key] = convert_dict_to_float(dict1[key])
		elif isinstance(dict1[key], np.float32):
			newdict[key] = float(dict1[key])
		else:
			newdict[key] = dict1[key]
	return newdict


def unroll_dict_with_list(input_dict: dict) -> dict:
	"""
	This function takes a dictionary containing numbers and lists and unrolls it into a flat dictionary.

	Args:
		input_dict (dict): the dictionary you would like to unroll

	Returns:
		dict: the unrolled dictionary
	"""
	newdict = {}
	for key in input_dict:
		if isinstance(input_dict[key], list) and isinstance(input_dict[key][0], list):
			for i, element in enumerate(input_dict[key]):
				newdict[f'{key}/vendor_{i}'] = element
		else:
			newdict[key] = input_dict[key]
	return newdict


def write_content_of_dict_to_overview_svg(
		manipulator: SVGManipulator,
		episode: int,
		episode_dictionary: dict,
		cumulated_dictionary: dict,
		config: AttrDict) -> None:
	"""
	This function takes a SVGManipulator and two dictionaries and translates the svg placeholder to the values in the dictionary

	Args:
		manipulator (SVGManipulator): instance of the class that can modify the Marketoverview_template.svg
		episode (int): current time step
		episode_dictionary (dict): monitoring dictionary of the current time step
		cumulated_dictionary (dict): monitoring dictionary with accumulated values for episodes
	"""
	episode += 1
	translated_dict = {
		'simulation_name': 'Market Simulation',
		'simulation_episode_length': str(config.episode_length),
		'simulation_current_episode': str(episode),
		'consumer_total_arrivals': str(episode * config.number_of_customers),
		'consumer_total_sales': str(episode * config.number_of_customers - cumulated_dictionary['customer/buy_nothing']),
		'a_competitor_name': 'vendor_0',
		'a_throw_away':	str(episode_dictionary['owner/throw_away']),
		'a_garbage': str(cumulated_dictionary['owner/throw_away']),
		'a_inventory': str(episode_dictionary['state/in_storage']['vendor_0']),
		'a_profit': '{0:.1f}'.format(cumulated_dictionary['profits/all']['vendor_0']),
		'a_price_new': str(episode_dictionary['actions/price_new']['vendor_0'] + 1),
		'a_price_used':	str(episode_dictionary['actions/price_refurbished']['vendor_0'] + 1),
		'a_rebuy_price': str(episode_dictionary['actions/price_rebuy']['vendor_0'] + 1),
		'a_repurchases': str(episode_dictionary['owner/rebuys']['vendor_0']),
		'a_resource_cost': str(config.production_price),
		'a_resources_in_use': str(episode_dictionary['state/in_circulation']),
		'a_sales_new': str(episode_dictionary['customer/purchases_new']['vendor_0']),
		'a_sales_used': str(episode_dictionary['customer/purchases_refurbished']['vendor_0']),
		'b_competitor_name': 'vendor_1',
		'b_inventory': str(episode_dictionary['state/in_storage']['vendor_1']),
		'b_profit': '{0:.1f}'.format(cumulated_dictionary['profits/all']['vendor_1']),
		'b_price_new': str(episode_dictionary['actions/price_new']['vendor_1'] + 1),
		'b_price_used': str(episode_dictionary['actions/price_refurbished']['vendor_1'] + 1),
		'b_rebuy_price': str(episode_dictionary['actions/price_rebuy']['vendor_1'] + 1),
		'b_repurchases': str(episode_dictionary['owner/rebuys']['vendor_1']),
		'b_resource_cost': str(config.production_price),
		'b_sales_new': str(episode_dictionary['customer/purchases_new']['vendor_1']),
		'b_sales_used': str(episode_dictionary['customer/purchases_refurbished']['vendor_1']),
	}

	manipulator.write_dict_to_svg(target_dictionary=translated_dict)


def get_class(import_string: str) -> object:
	"""
	Get the class from the given string.

	Args:
		import_string (str): A string containing the import path in the format 'module.submodule.class'.

	Returns:
		A class object: The imported class.
	"""
	module_name, class_name = import_string.rsplit('.', 1)
	try:
		return getattr(importlib.import_module(module_name), class_name)
	except AttributeError as error:
		raise AttributeError(f'The string you passed could not be resolved to a class: {import_string}') from error
	except ModuleNotFoundError as error:
		raise ModuleNotFoundError(f'The string you passed could not be resolved to a module: {import_string}') from error


def filtered_class_str_from_dir(import_path: str, all_classes: list, regex_match: str) -> list:
	"""
	Filters all given class strings by regex and returns the classes concatinated with the import path as list.s

	Args:
		import_path (str): Path the class is imported from, i.e. 'recommerce...'
		all_classes (list): all classes, that should be filtered
		regex_match (str): The regex that should be used for filtering

	Returns:
		list: list of filtered class strings starting with the import path
	"""
	filtered_classes = list(set(filter(lambda class_name: re.match(regex_match, class_name), all_classes)))
	return [import_path + '.' + f_class for f_class in sorted(filtered_classes)]
