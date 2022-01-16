import copy
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.utils as ut
import market.sim_market as sim_market
from monitoring.svg_manipulation import SVGManipulator


def run_example(marketplace=sim_market.CircularEconomyRebuyPriceOneCompetitor(), agent=vendors.RuleBasedCERebuyAgent(), log_dir_prepend='') -> int:
	"""
	Run a specified marketplace with a (pre-trained, if RL) agent and record various statistics using TensorBoard.

	Args:
		marketplace (sim_market instance, optional): The market environment to run the simulation on. Defaults to sim_market.CircularEconomyRebuyPriceOneCompetitor().
		agent (agent instance, optional): The agent to run the simulation on. Defaults to vendors.RuleBasedCERebuyAgent().
		log_dir_prepend (str, optional): What to prepend to the log_dir folder name. Defaults to ''.

	Returns:
		int: The profit made.
	"""
	counter = 0
	our_profit = 0
	is_done = False
	state = marketplace.reset()

	signature = f'{log_dir_prepend}exampleprinter_{time.strftime("%b%d_%H-%M-%S")}'
	writer = SummaryWriter(log_dir=os.path.join('results', 'runs', signature))

	if isinstance(marketplace, sim_market.CircularEconomyRebuyPriceOneCompetitor):
		svg_manipulator = SVGManipulator(signature)
	cumulative_dict = None

	with torch.no_grad():
		while not is_done:
			action = agent.policy(state)
			print(state)
			state, reward, is_done, logdict = marketplace.step(action)
			if cumulative_dict is not None:
				cumulative_dict = ut.add_content_of_two_dicts(cumulative_dict, logdict)
			else:
				cumulative_dict = copy.deepcopy(logdict)
			ut.write_dict_to_tensorboard(writer, logdict, counter)
			ut.write_dict_to_tensorboard(writer, cumulative_dict, counter, is_cumulative=True)
			if isinstance(marketplace, sim_market.CircularEconomyRebuyPriceOneCompetitor):
				ut.write_content_of_dict_to_overview_svg(svg_manipulator, counter, logdict, cumulative_dict)
			our_profit += reward
			counter += 1
			if isinstance(marketplace, sim_market.CircularEconomyRebuyPriceOneCompetitor):
				svg_manipulator.save_overview_svg(filename=('MarketOverview_%.3d' % counter))

	if isinstance(marketplace, sim_market.CircularEconomyRebuyPriceOneCompetitor):
		svg_manipulator.to_html()

	return our_profit


if __name__ == '__main__':  # pragma: no cover
	print(run_example())
