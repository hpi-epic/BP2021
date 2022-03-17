import os
import signal
import sys
from copy import deepcopy

import monitoring.agent_monitoring.am_configuration as am_configuration
import monitoring.agent_monitoring.am_evaluation as am_evaluation
from configuration.environment_config import AgentMonitoringEnvironmentConfig, EnvironmentConfigLoader


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

		# initialize the rewards list with a list for each agent
		rewards = [[] for _ in range(len(self.configurator.agents))]

		for episode in range(1, self.configurator.episodes + 1):
			# reset the state & marketplace once to be used by all agents
			source_state = self.configurator.marketplace.reset()
			source_marketplace = self.configurator.marketplace

			for i in range(len(self.configurator.agents)):
				# for every agent, set an equivalent "start-market"
				self.configurator.marketplace = deepcopy(source_marketplace)

				# reset values for all agents
				state = source_state
				episode_reward = 0
				is_done = False

				# run marketplace for this agent
				while not is_done:
					action = self.configurator.agents[i].policy(state)
					state, step_reward, is_done, _ = self.configurator.marketplace.step(action)
					episode_reward += step_reward

				# removing this will decrease our performance when we still want to do live drawing
				# could think about a caching strategy for live drawing
				# add the reward to the current agent's reward-Array
				rewards[i] += [episode_reward]

			# after all agents have run the episode
			if (episode % 100) == 0:
				print(f'Running {episode}th episode...')

			if (episode % self.configurator.plot_interval) == 0:
				self.evaluator.create_histogram(rewards, f'episode_{episode}')

		# if the plot_interval does not create a plot after the last episode automatically, we will do it manually
		if (self.configurator.episodes % self.configurator.plot_interval) != 0:
			self.evaluator.create_histogram(rewards, f'episode_{self.configurator.episodes}')

		return rewards


def run_monitoring_session(monitor: Monitor = Monitor()) -> None:
	"""
	Run a monitoring session with a configured Monitor() and display and save metrics.

	Args:
		monitor (Monitor instance, optional): The monitor to run the session on. Defaults to a default Monitor() instance.
	"""
	monitor.configurator.print_configuration()

	print('\nStarting monitoring session...')
	rewards = monitor.run_marketplace()

	monitor.evaluator.evaluate_session(rewards)


if __name__ == '__main__':  # pragma: no cover
	monitor = Monitor()
	config: AgentMonitoringEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_agent_monitoring')
	monitor.configurator.setup_monitoring(
		enable_live_draw=config.enable_live_draw,
		episodes=config.episodes,
		plot_interval=config.plot_interval,
		marketplace=config.marketplace,
		agents=config.agent
	)
	run_monitoring_session(monitor)
