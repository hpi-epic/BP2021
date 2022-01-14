import copy
import signal
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.utils as ut
import market.sim_market as sim_market
from monitoring.svg_manipulation import SVGManipulator


class ExamplePrinter():

	def __init__(self, environment=sim_market.CircularEconomyRebuyPriceOneCompetitor(), agent=vendors.RuleBasedCERebuyAgent()):
		self.environment = environment
		self.agent = agent
		# Signal handler for e.g. KeyboardInterrupt
		signal.signal(signal.SIGINT, self._signal_handler)

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
			env (sim_market instance, optional): The market environment to run the simulation on. Defaults to sim_market.CircularEconomyRebuyPriceOneCompetitor().
			agent (agent instance, optional): The agent to run the simulation on. Defaults to vendors.RuleBasedCERebuyAgent().
			log_dir_prepend (str, optional): What to prepend to the log_dir folder name. Defaults to ''.

		Returns:
			int: The profit made.
		"""
		counter = 0
		our_profit = 0
		is_done = False
		state = self.environment.reset()
		signature = time.strftime('%Y%m%d-%H%M%S') + f'_{type(self.environment).__name__}_{type(self.agent).__name__}_exampleprinter'
		# writer = SummaryWriter(log_dir='runs/' + log_dir_prepend + signature)
		writer = SummaryWriter()
		if isinstance(self.environment, sim_market.CircularEconomyRebuyPriceOneCompetitor):
			svg_manipulator = SVGManipulator(signature)
		cumulative_dict = None

		with torch.no_grad():
			while not is_done:
				action = self.agent.policy(state)
				print(state)
				state, reward, is_done, logdict = self.environment.step(action)
				if cumulative_dict is not None:
					cumulative_dict = ut.add_content_of_two_dicts(cumulative_dict, logdict)
				else:
					cumulative_dict = copy.deepcopy(logdict)
				ut.write_dict_to_tensorboard(writer, logdict, counter)
				ut.write_dict_to_tensorboard(writer, cumulative_dict, counter, is_cumulative=True)
				if isinstance(self.environment, sim_market.CircularEconomyRebuyPriceOneCompetitor):
					ut.write_content_of_dict_to_overview_svg(svg_manipulator, counter, logdict, cumulative_dict)
				our_profit += reward
				counter += 1
				if isinstance(self.environment, sim_market.CircularEconomyRebuyPriceOneCompetitor):
					svg_manipulator.save_overview_svg(filename=('MarketOverview_%.3d' % counter))

		if isinstance(self.environment, sim_market.CircularEconomyRebuyPriceOneCompetitor):
			svg_manipulator.to_html()

		return our_profit


if __name__ == '__main__':  # pragma: no cover
	print(ExamplePrinter().run_example())
