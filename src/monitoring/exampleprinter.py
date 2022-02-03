import copy
import os
import signal
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.utils as ut
import market.circular.circular_sim_market as circular_market
from monitoring.svg_manipulation import SVGManipulator


class ExamplePrinter():

	def __init__(self):
		ut.ensure_results_folders_exist()
		self.marketplace = circular_market.CircularEconomyRebuyPriceOneCompetitor()
		self.agent = vendors.RuleBasedCERebuyAgent()
		# Signal handler for e.g. KeyboardInterrupt
		signal.signal(signal.SIGINT, self._signal_handler)

	def setup_exampleprinter(self, marketplace=None, agent=None):
		"""
		Configure the current exampleprinter session.

		Args:
			marketplace (SimMarket instance, optional): What marketplace to run the session on.
			agent (Agent instance, optional): What agent ot run the session on..
		"""
		if(marketplace is not None):
			self.marketplace = marketplace
		if(agent is not None):
			self.agent = agent

	def _signal_handler(self, signum, frame):  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		print('\nAborting exampleprinter run...')
		sys.exit(0)

	def run_example(self, log_dir_prepend='') -> int:
		"""
		Run a specified marketplace with a (pre-trained, if RL) agent and record various statistics using TensorBoard.

		Args:
			log_dir_prepend (str, optional): What to prepend to the log_dir folder name. Defaults to ''.

		Returns:
			int: The profit made.
		"""
		counter = 0
		our_profit = 0
		is_done = False
		state = self.marketplace.reset()

		signature = f'{log_dir_prepend}exampleprinter_{time.strftime("%b%d_%H-%M-%S")}'
		writer = SummaryWriter(log_dir=os.path.join('results', 'runs', signature))

		if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceOneCompetitor):
			svg_manipulator = SVGManipulator(signature)
		cumulative_dict = None

		with torch.no_grad():
			while not is_done:
				action = self.agent.policy(state)
				print(state)
				state, reward, is_done, logdict = self.marketplace.step(action)
				if cumulative_dict is not None:
					cumulative_dict = ut.add_content_of_two_dicts(cumulative_dict, logdict)
				else:
					cumulative_dict = copy.deepcopy(logdict)
				ut.write_dict_to_tensorboard(writer, logdict, counter)
				ut.write_dict_to_tensorboard(writer, cumulative_dict, counter, is_cumulative=True)
				if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceOneCompetitor):
					ut.write_content_of_dict_to_overview_svg(svg_manipulator, counter, logdict, cumulative_dict)
				our_profit += reward
				counter += 1
				if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceOneCompetitor):
					svg_manipulator.save_overview_svg(filename=('MarketOverview_%.3d' % counter))

		if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceOneCompetitor):
			svg_manipulator.to_html()

		return our_profit


if __name__ == '__main__':  # pragma: no cover
	print(ExamplePrinter().run_example())
