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
	"""
	This function takes a dictionary of data with data from one step and adds it at the specified time to the tensorboard.

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
			assert isinstance(dict1[key], int) or isinstance(dict1[key], float), 'the dictionary should only contain numbers (int or float)'
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
			assert isinstance(dict1[key], int) or isinstance(dict1[key], float), 'dict1 should only contain numbers (int or float)'
			assert isinstance(dict2[key], int) or isinstance(dict2[key], float), 'dict2 should only contain numbers (int or float)'
			newdict[key] = dict1[key] + dict2[key]
	return newdict


def write_content_of_dict_to_overview_svg(manipulator, episode, episode_dictionary, cumulated_dictionary) -> None:
	"""
	This function takes a SVGManipulator and two dictionaries and translates the svg placeholder to the values in the dictionary

	Args:
		manipulator (SVGManipulator): instance of the class that can modify the Marketoverview_template.svg
		episode (int): current time step
		episode_dictionary (dict): monitoring dictionary of the current time step
		cumulated_dictionary (dict): monitoring dictionary with accumulated values for episodes
	"""
	episode += 1
	translated_dict = {}
	translated_dict['simulation_name'] = 'Market Simulation'
	translated_dict['simulation_episode_length'] = str(config.EPISODE_LENGTH)
	translated_dict['simulation_current_episode'] = str(episode)
	translated_dict['consumer_total_arrivals'] = str(episode * config.NUMBER_OF_CUSTOMERS)
	translated_dict['consumer_total_sales'] = str(episode * config.NUMBER_OF_CUSTOMERS - cumulated_dictionary['customer/buy_nothing'])
	translated_dict['a_competitor_name'] = 'vendor_0'
	translated_dict['a_throw_away'] = str(episode_dictionary['owner/throw_away'])
	translated_dict['a_garbage'] = str(cumulated_dictionary['owner/throw_away'])
	translated_dict['a_inventory'] = str(episode_dictionary['state/in_storage']['vendor_0'])
	translated_dict['a_profit'] = str(cumulated_dictionary['profits/all']['vendor_0'])
	translated_dict['a_price_new'] = str(episode_dictionary['actions/price_new']['vendor_0'] + 1)
	translated_dict['a_price_used'] = str(episode_dictionary['actions/price_refurbished']['vendor_0'] + 1)
	translated_dict['a_rebuy_price'] = str(episode_dictionary['actions/price_rebuy']['vendor_0'] + 1)
	translated_dict['a_repurchases'] = str(episode_dictionary['owner/rebuys']['vendor_0'])
	translated_dict['a_resource_cost'] = str(config.PRODUCTION_PRICE)
	translated_dict['a_resources_in_use'] = str(episode_dictionary['state/in_circulation'])
	translated_dict['a_sales_new'] = str(episode_dictionary['customer/purchases_new']['vendor_0'])
	translated_dict['a_sales_used'] = str(episode_dictionary['customer/purchases_refurbished']['vendor_0'])
	translated_dict['b_competitor_name'] = 'vendor_1'
	translated_dict['b_inventory'] = str(episode_dictionary['state/in_storage']['vendor_1'])
	translated_dict['b_profit'] = str(cumulated_dictionary['profits/all']['vendor_1'])
	translated_dict['b_price_new'] = str(episode_dictionary['actions/price_new']['vendor_1'] + 1)
	translated_dict['b_price_used'] = str(episode_dictionary['actions/price_refurbished']['vendor_1'] + 1)
	translated_dict['b_rebuy_price'] = str(episode_dictionary['actions/price_rebuy']['vendor_1'] + 1)
	translated_dict['b_repurchases'] = str(episode_dictionary['owner/rebuys']['vendor_1'])
	translated_dict['b_resource_cost'] = str(config.PRODUCTION_PRICE)
	translated_dict['b_sales_new'] = str(episode_dictionary['customer/purchases_new']['vendor_1'])
	translated_dict['b_sales_used'] = str(episode_dictionary['customer/purchases_refurbished']['vendor_1'])
	manipulator.write_dict_to_svg(target_dictionary=translated_dict)