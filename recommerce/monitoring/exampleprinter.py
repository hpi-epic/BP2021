import copy
import os
import signal
import sys
import time

import torch
from attrdict import AttrDict
from torch.utils.tensorboard import SummaryWriter

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
from recommerce.configuration.environment_config import EnvironmentConfigLoader, ExampleprinterEnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent
from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import Agent
from recommerce.monitoring.svg_manipulation import SVGManipulator
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO


class ExamplePrinter():

	def __init__(self, config_market: AttrDict):
		ut.ensure_results_folders_exist()
		self.config_market = config_market
		self.marketplace = circular_market.CircularEconomyRebuyPriceDuopoly(config=self.config_market)
		self.agent = RuleBasedCERebuyAgent(config_market=self.config_market)
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
				print(action)
				state, reward, is_done, logdict = self.marketplace.step(action)
				if cumulative_dict is not None:
					cumulative_dict = ut.add_content_of_two_dicts(cumulative_dict, logdict)
				else:
					cumulative_dict = copy.deepcopy(logdict)
				ut.write_dict_to_tensorboard(writer, logdict, counter)
				ut.write_dict_to_tensorboard(writer, cumulative_dict, counter, is_cumulative=True,
					episode_length=self.marketplace.config.episode_length)
				if isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceDuopoly):
					ut.write_content_of_dict_to_overview_svg(svg_manipulator, counter, logdict, cumulative_dict, self.config_market)
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
	config_environment: ExampleprinterEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_exampleprinter')

	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
	config_rl: AttrDict = HyperparameterConfigLoader.load('q_learning_config', config_environment.agent[0]['agent_class'])
	printer = ExamplePrinter(config_market=config_market)

	# TODO: Theoretically, the name of the agent is saved in config_environment['name'], but we don't use it yet.
	marketplace = config_environment.marketplace(config=config_market, competitors=config_environment.agent[1:])

	# QLearningAgents need more initialization
	if issubclass(config_environment.agent[0]['agent_class'], QLearningAgent):
		printer.setup_exampleprinter(marketplace=marketplace,
			agent=config_environment.agent[0]['agent_class'](
				config_market=config_market,
				config_rl=config_rl,
				marketplace=marketplace,
				load_path=os.path.abspath(os.path.join(PathManager.data_path, config_environment.agent[0]['argument']))))
	else:
		printer.setup_exampleprinter(marketplace=marketplace, agent=config_environment.agent[0]['agent_class']())

	print(f'The final profit was: {printer.run_example()}')


if __name__ == '__main__':  # pragma: no cover
	# Make sure a valid datapath is set
	PathManager.manage_user_path()

	config_market = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceDuopoly)
	config_rl = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
	market = circular_market.CircularEconomyRebuyPriceDuopoly(config_market, support_continuous_action_space=True)
	agent = StableBaselinesPPO(config_market, config_rl, market,
		'C:\\Users\\jangr\\develop\\BP2021\\results\\trainedModels\\Stable_Baselines_PPO_Jun12_15-20-30\\Stable_Baselines_PPO_00500.zip')
	printer = ExamplePrinter(config_market)
	printer.setup_exampleprinter(market, agent)
	printer.run_example()

	# main()
