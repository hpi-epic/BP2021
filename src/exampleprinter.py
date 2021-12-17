#!/usr/bin/env python3

import copy

import torch
from torch.utils.tensorboard import SummaryWriter

import agent as a
import sim_market


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
			newdict[key] = dict1[key] + dict2[key]
	return newdict


def print_example(env=sim_market.CircularEconomy(), agent=a.RuleBasedCEAgent()) -> int:
	counter = 0
	our_profit = 0
	is_done = False
	state = env.reset()
	writer = SummaryWriter()
	cumulative_dict = None

	with torch.no_grad():
		while not is_done:
			action = agent.policy(state)
			print(state)
			state, reward, is_done, logdict = env.step(action)
			if cumulative_dict is not None:
				cumulative_dict = add_content_of_two_dicts(cumulative_dict, logdict)
			else:
				cumulative_dict = copy.deepcopy(logdict)
			write_dict_to_tensorboard(writer, logdict, counter)
			write_dict_to_tensorboard(writer, cumulative_dict, counter, is_cumulative=True)
			our_profit += reward
			counter += 1
	return our_profit


def main() -> None:
	agent = a.RuleBasedCERebuyAgent()
	environment = sim_market.CircularEconomyRebuyPrice()
	print_example(environment, agent)


if __name__ == '__main__':
	main()
