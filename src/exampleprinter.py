#!/usr/bin/env python3

import copy

import torch
from torch.utils.tensorboard import SummaryWriter

import agent as a
import sim_market


def write_dict_to_tensorboard(writer, mydict, counter, cum=False):
	for name, content in mydict.items():
		if cum and (name.startswith('actions') or name.startswith('state')):
			continue
		if cum:
			name = 'cumulated_' + name
		if isinstance(content, dict):
			writer.add_scalars(name, content, counter)
		else:
			writer.add_scalar(name, content, counter)


def add_content_of_two_dicts(dict1, dict2) -> dict:
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
	cumdict = None

	with torch.no_grad():
		while not is_done:
			action = agent.policy(state)
			print(state)
			state, reward, is_done, mydict = env.step(action)
			if cumdict is not None:
				cumdict = add_content_of_two_dicts(cumdict, mydict)
			else:
				cumdict = copy.deepcopy(mydict)
			write_dict_to_tensorboard(writer, mydict, counter)
			write_dict_to_tensorboard(writer, cumdict, counter, cum=True)
			our_profit += reward
			counter += 1
	return our_profit


if __name__ == '__main__':
	agent = a.RuleBasedCERebuyAgent()
	environment = sim_market.CircularEconomyRebuyPrice()
	print_example(environment, agent)
