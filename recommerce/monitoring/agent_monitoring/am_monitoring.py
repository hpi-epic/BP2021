import os
import signal
import sys
from copy import deepcopy

import torch
from attrdict import AttrDict
from tqdm import trange

import recommerce.monitoring.agent_monitoring.am_configuration as am_configuration
import recommerce.monitoring.agent_monitoring.am_evaluation as am_evaluation
from recommerce.configuration.environment_config import AgentMonitoringEnvironmentConfig, EnvironmentConfigLoader
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.monitoring.watcher import Watcher

print('successfully imported torch: cuda?', torch.cuda.is_available())


class Monitor():
	"""
	A Monitor() monitors given agents on a marketplace, recording the rewards achieved by the agents.

	When the run is finished, diagrams will be created in the 'results/monitoring' folder by the Evaluator. \\
	The Monitor() can be customized using its Configurator() with configurator.setup_monitoring().
	"""
	def __init__(self):
		self.configurator = am_configuration.Configurator()
		self.evaluator = am_evaluation.Evaluator(self.configurator)
		# Signal handler for e.g. KeyboardInterrupt
		signal.signal(signal.SIGINT, self._signal_handler)

	def _signal_handler(self, signum, frame):  # pragma: no cover
		"""
		Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
		"""
		print('\nAborting monitoring session...')
		print(f'All histograms were saved to {os.path.abspath(self.configurator.folder_path)}')
		sys.exit(0)

	def run_marketplace(self) -> list:
		"""
		Run the marketplace with the given monitoring configuration.

		Does not create any diagrams.

		Returns:
			list: A list with a list of rewards for each agent.
		"""
		# each agent on its own marketplace
		if self.configurator.separate_markets:
			# initialize the watcher list with a list for each agent
			watchers = [Watcher(config_market=self.configurator.marketplace.config) for _ in range(len(self.configurator.agents))]

			for _ in trange(1, self.configurator.episodes + 1, unit=' episodes', leave=False):
				# reset the state & marketplace once to be used by all agents
				source_state = self.configurator.marketplace.reset()
				source_marketplace = self.configurator.marketplace

				for current_agent_index in range(len(self.configurator.agents)):
					# for every agent, set an equivalent "start-market"
					self.configurator.marketplace = deepcopy(source_marketplace)
					state = source_state
					is_done = False

					# run marketplace for this agent
					while not is_done:
						action = self.configurator.agents[current_agent_index].policy(state)
						state, _, is_done, info = self.configurator.marketplace.step(action)
						watchers[current_agent_index].add_info(info)

			return [watcher.get_cumulative_properties() for watcher in watchers]

		# all agents on one marketplace
		else:
			watcher = Watcher(config_market=self.configurator.marketplace.config)
			for _ in trange(1, self.configurator.episodes + 1, unit=' episodes', leave=False):
				state = self.configurator.marketplace.reset()
				is_done = False

				# run marketplace for all agents
				while not is_done:
					action = self.configurator.agents[0].policy(state)
					state, _, is_done, info = self.configurator.marketplace.step(action)
					watcher.add_info(info)

			return watcher.get_cumulative_properties()


def run_monitoring_session(monitor: Monitor) -> None:
	"""
	Run a monitoring session with a configured Monitor() and display and save metrics.

	Args:
		monitor (Monitor instance, optional): The monitor to run the session on. Defaults to a default Monitor() instance.
	"""
	monitor.configurator.print_configuration()

	if monitor.configurator.separate_markets:
		print('\nAgents are playing on separate markets...')
	else:
		print('\nAgents are playing on the same market...')
	print('\nStarting monitoring session...')
	rewards = monitor.run_marketplace()

	monitor.evaluator.evaluate_session(rewards)


def main():  # pragma: no cover
	"""
	Defines what is performed when the `agent_monitoring` command is chosen in `main.py`.
	"""
	monitor = Monitor()
	config_environment_am: AgentMonitoringEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_agent_monitoring')
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment_am.marketplace)
	monitor.configurator.setup_monitoring(
		episodes=config_environment_am.episodes,
		plot_interval=config_environment_am.plot_interval,
		marketplace=config_environment_am.marketplace,
		agents=config_environment_am.agent,
		separate_markets=config_environment_am.separate_markets,
		config_market=config_market,
		support_continuous_action_space=True
	)
	run_monitoring_session(monitor)


if __name__ == '__main__':  # pragma: no cover
	# Make sure a valid datapath is set
	PathManager.manage_user_path()

	main()
