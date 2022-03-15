import copy
import os
import signal
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import configuration.utils as ut
import market.circular.circular_sim_market as circular_market
from agents.vendors import Agent
from configuration.environment_config import EnvironmentConfigLoader, ExampleprinterEnvironmentConfig
from market.circular.circular_vendors import RuleBasedCERebuyAgent
from market.sim_market import SimMarket
from monitoring.svg_manipulation import SVGManipulator
from rl.q_learning_agent import QLearningAgent


class ExamplePrinter():

	def __init__(self):
		ut.ensure_results_folders_exist()
		self.marketplace = circular_market.CircularEconomyRebuyPriceOneCompetitor()
		self.agent = RuleBasedCERebuyAgent()
		# Signal handler for e.g. KeyboardInterrupt
		signal.signal(signal.SIGINT, self._signal_handler)

	def setup_exampleprinter(self, marketplace: SimMarket = None, agent: Agent = None) -> None:
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

	def _signal_handler(self, signum, frame) -> None:  # pragma: no cover
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
		print(f'Running exampleprinter on a {self.marketplace.__class__.__name__} market with a {self.agent.__class__.__name__} agent...')
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
	printer = ExamplePrinter()

	config: ExampleprinterEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_exampleprinter')
	marketplace = config.marketplace()

	# QLearningAgents need more initialization
	if issubclass(config.agent[0], QLearningAgent):
		printer.setup_exampleprinter(marketplace=marketplace,
			agent=config.agent[0](
				n_observations=marketplace.observation_space.shape[0],
				n_actions=marketplace.get_n_actions(),
				load_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', config.agent[1]))))
	else:
		printer.setup_exampleprinter(marketplace=marketplace, agent=config.agent[0]())

	print(f'The final profit was: {printer.run_example()}')
