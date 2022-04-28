import copy
import os
import signal
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
from recommerce.configuration.environment_config import EnvironmentConfigLoader, ExampleprinterEnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader, HyperparameterConfig
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent
from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import Agent
from recommerce.monitoring.svg_manipulation import SVGManipulator
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent


class ExamplePrinter():

	def __init__(self, config: HyperparameterConfig):
		ut.ensure_results_folders_exist()
		self.config = config
		self.marketplace = circular_market.CircularEconomyRebuyPriceDuopoly(config=self.config)
		self.agent = RuleBasedCERebuyAgent(config=self.config)
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

	def run_example(self) -> int:
		"""
		Run a specified marketplace with a (pre-trained, if RL) agent and record various statistics using TensorBoard.

		Returns:
			int: The profit made.
		"""
		print(f'Running exampleprinter on a {self.marketplace.__class__.__name__} market with a {self.agent.__class__.__name__} agent...')
		counter = 0
		our_profit = 0
		is_done = False
		state = self.marketplace.reset()

		signature = f'exampleprinter_{time.strftime("%b%d_%H-%M-%S")}'
		writer = SummaryWriter(log_dir=os.path.join(PathManager.results_path, 'runs', signature))

		if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceDuopoly):
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
				if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceDuopoly):
					ut.write_content_of_dict_to_overview_svg(svg_manipulator, counter, logdict, cumulative_dict, self.config)
				our_profit += reward
				counter += 1
				if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceDuopoly):
					svg_manipulator.save_overview_svg(filename=('MarketOverview_%.3d' % counter))

		if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceDuopoly):
			svg_manipulator.to_html()

		return our_profit


def main():  # pragma: no cover
	"""
	Defines what is performed when the `agent_monitoring` command is chosen in `main.py`.
	"""
	config_hyperparameter: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')
	printer = ExamplePrinter(config_hyperparameter)

	config: ExampleprinterEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_exampleprinter')
	# TODO: Theoretically, the name of the agent is saved in config['name'], but we don't use it yet.
	marketplace = config.marketplace()

	# QLearningAgents need more initialization
	if issubclass(config.agent['agent_class'], QLearningAgent):
		printer.setup_exampleprinter(marketplace=marketplace,
			agent=config.agent['agent_class'](
				marketplace=marketplace,
				load_path=os.path.abspath(os.path.join(PathManager.data_path, config.agent['argument']))))
	else:
		printer.setup_exampleprinter(marketplace=marketplace, agent=config.agent['agent_class']())

	print(f'The final profit was: {printer.run_example()}')


if __name__ == '__main__':  # pragma: no cover
	main()
