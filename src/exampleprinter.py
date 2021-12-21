#!/usr/bin/env python3

import copy

import torch
from torch.utils.tensorboard import SummaryWriter

import agent as a
import sim_market
import utils as ut


def print_example(env=sim_market.CircularEconomyMonopolyScenario(), agent=a.RuleBasedCEAgent()) -> int:
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
				cumulative_dict = ut.add_content_of_two_dicts(cumulative_dict, logdict)
			else:
				cumulative_dict = copy.deepcopy(logdict)
			ut.write_dict_to_tensorboard(writer, logdict, counter)
			ut.write_dict_to_tensorboard(writer, cumulative_dict, counter, is_cumulative=True)
			our_profit += reward
			counter += 1
	return our_profit


def main() -> None:  # pragma: no cover
	agent = a.RuleBasedCERebuyAgent()
	environment = sim_market.CircularEconomyRebuyPrice()
	print_example(environment, agent)


if __name__ == '__main__':  # pragma: no cover
	main()
