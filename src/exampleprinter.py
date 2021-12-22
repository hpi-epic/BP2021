#!/usr/bin/env python3

import copy
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import sim_market
import utils_sim_market as ut
import vendors


def print_example(environment=sim_market.CircularEconomy(), agent=vendors.RuleBasedCEAgent(), log_dir_prepend='') -> int:
	"""
	Run a specified marketplace with a (pre-trained, if RL) agent and record various statistics using TensorBoard.

	Args:
		env (sim_market instance, optional): The market environment to run the simulation on. Defaults to sim_market.CircularEconomy().
		agent (agent instance, optional): The agent to run the simulation on. Defaults to a.RuleBasedCEAgent().
		log_dir_prepend (str, optional): What to prepend to the log_dir folder name. Defaults to ''.

	Returns:
		int: The profit made.
	"""
	counter = 0
	our_profit = 0
	is_done = False
	state = environment.reset()
	writer = SummaryWriter(log_dir='runs/' + log_dir_prepend + time.strftime('%Y%m%d-%H%M%S') + f'_{type(environment).__name__}_{type(agent).__name__}_exampleprinter')
	cumulative_dict = None

	with torch.no_grad():
		while not is_done:
			action = agent.policy(state)
			print(state)
			state, reward, is_done, logdict = environment.step(action)
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
	used_agent = vendors.RuleBasedCERebuyAgent()
	environment = sim_market.CircularEconomyRebuyPrice()
	print_example(environment, used_agent)


if __name__ == '__main__':  # pragma: no cover
	main()
