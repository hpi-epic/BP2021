import os
import signal
import sys
from copy import deepcopy

import torch
from tqdm import trange

import recommerce.monitoring.agent_monitoring.am_configuration as am_configuration
import recommerce.monitoring.agent_monitoring.am_evaluation as am_evaluation
from recommerce.configuration.environment_config import AgentMonitoringEnvironmentConfig, EnvironmentConfigLoader
from recommerce.configuration.hyperparameter_config import HyperparameterConfig, HyperparameterConfigLoader
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

		Automatically produces histograms, but not metric diagrams.

		Returns:
			list: A list with a list of rewards for each agent
		"""
		config = HyperparameterConfigLoader.load('hyperparameter_config')
		# initialize the watcher list with a list for each agent
		watchers = [Watcher(config=config) for _ in range(len(self.configurator.agents))]

		for episode in trange(1, self.configurator.episodes + 1, unit=' episodes', leave=False):
			# reset the state & marketplace once to be used by all agents
			source_state = self.configurator.marketplace.reset()
			source_marketplace = self.configurator.marketplace

			for current_agent_index in range(len(self.configurator.agents)):
				# for every agent, set an equivalent "start-market"
				self.configurator.marketplace = deepcopy(source_marketplace)

				# reset values for all agents
				state = source_state
				is_done = False

				# run marketplace for this agent
				while not is_done:
					action = self.configurator.agents[current_agent_index].policy(state)
					state, _, is_done, info = self.configurator.marketplace.step(action)
					watchers[current_agent_index].add_info(info)

		# only one histogram after the whole monitoring process
		returns = [watcher.get_all_samples_of_property('profits/all', 0) for watcher in watchers]
		self.evaluator.create_histogram(returns, True, 'Cumulative_rewards_per_episode.svg')

		return [watcher.get_cumulative_properties() for watcher in watchers]


def run_monitoring_session(monitor: Monitor) -> None:
	"""
	Run a monitoring session with a configured Monitor() and display and save metrics.

	Args:
		monitor (Monitor instance, optional): The monitor to run the session on. Defaults to a default Monitor() instance.
	"""
	monitor.configurator.print_configuration()

	print('\nStarting monitoring session...')
	rewards = monitor.run_marketplace()

	monitor.evaluator.evaluate_session(rewards)


def main():  # pragma: no cover
	"""
	Defines what is performed when the `agent_monitoring` command is chosen in `main.py`.
	"""
	monitor = Monitor()
	config_environment_am: AgentMonitoringEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_agent_monitoring')
	config_hyperparameter: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')
	monitor.configurator.setup_monitoring(
		enable_live_draw=config_environment_am.enable_live_draw,
		episodes=config_environment_am.episodes,
		plot_interval=config_environment_am.plot_interval,
		marketplace=config_environment_am.marketplace,
		agents=config_environment_am.agent,
		config=config_hyperparameter
	)
	run_monitoring_session(monitor)


if __name__ == '__main__':  # pragma: no cover
	# Make sure a valid datapath is set
	PathManager.manage_user_path()

	main()
