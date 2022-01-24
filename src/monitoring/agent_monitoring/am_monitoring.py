import os
import signal
import sys

import rl.actorcritic_agent as actorcritic_agent

import monitoring.agent_monitoring.am_configuration as am_configuration
import monitoring.agent_monitoring.am_evaluation as am_evaluation

import agents.vendors as vendors
import market.sim_market as sim_market


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
			# reset the state once to be used by all agents
			default_state = self.configurator.marketplace.reset()

			for i in range(len(self.configurator.agents)):
				# reset marketplace, bit hacky, if you find a better solution feel free
				self.configurator.marketplace.reset()
				self.configurator.marketplace.state = default_state

				# reset values for all agents
				state = default_state
				episode_reward = 0
				is_done = False

				# run marketplace for this agent
				while not is_done:
					action = self.configurator.agents[i].policy(state)[0].tolist() if i == 0 else self.configurator.agents[i].policy(state)
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
				self.evaluator.create_histogram(rewards, 'episode_' + str(episode))
		return rewards


def run_monitoring_session(monitor: Monitor = Monitor()) -> None:
	"""
	Run a monitoring session with a configured Monitor() and display and save metrics.

	Args:
		monitor (Monitor instance, optional): The monitor to run the session on. Defaults to a default Monitor() instance.
	"""
	mymarket = sim_market.CircularEconomyRebuyPriceOneCompetitor()

	myagent = actorcritic_agent.ContinuosActorCriticAgentFixedOneStd(mymarket.observation_space.shape[0], 3)
	myagent.load_actor('results\\monitoring\\actor_parametersCircularEconomyRebuyPriceOneCompetitor_ContinuosActorCriticAgentFixedOneStd_650.340.dat')

	monitor.configurator.agents = [myagent, vendors.RuleBasedCERebuyAgent()]
	monitor.configurator.marketplace = mymarket

	monitor.configurator.agent_colors = [(0, 1, 1, 1), (1, 0, 1, 1)]
	# monitor.configurator.print_configuration()

	print('\nStarting monitoring session...')
	rewards = monitor.run_marketplace()

	monitor.evaluator.evaluate_session(rewards)


if __name__ == '__main__':  # pragma: no cover
	run_monitoring_session()
