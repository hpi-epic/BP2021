import os
import time

import matplotlib.pyplot as plt
import numpy as np

import agent
import sim_market as sim


class Monitor():

	def __init__(self) -> None:
		self.enable_live_draws = True
		self.episodes = 500
		self.histogram_plot_interval = int(self.episodes / 10)
		self.path_to_modelfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + 'test_marketplace.dat'
		self.situation = 'linear'
		self.marketplace = sim.CircularEconomy() if self.situation == 'circular' else sim.ClassicScenario()
		self.agents = [agent.QLearningAgent(self.marketplace.observation_space.shape[0], self.marketplace.action_space.n, load_path=self.path_to_modelfile)]
		self.agent_colors = ['#0000ff']

	# helper functions
	def round_up(self, number, decimals=0):
		multiplier = 10 ** decimals
		return np.ceil(number * multiplier) / multiplier

	def get_cmap(self, n, name='hsv'):
		return plt.cm.get_cmap(name, n + 1)

	# configure the situation to be monitored
	def setup_monitoring(self, draw_enabled=None, new_episodes=None, new_interval=None, new_modelfile=None, new_situation=None, new_marketplace=None, new_agents=None) -> None:
		# doesn't look nice, but afaik only way to keep parameter list short
		if(draw_enabled is not None):
			self.enable_live_draws = draw_enabled
		if(new_episodes is not None):
			self.episodes = new_episodes
		if(new_interval is not None):
			self.histogram_plot_interval = new_interval
		if(new_modelfile is not None):
			self.path_to_modelfile = new_modelfile
		if(new_situation is not None):
			self.situation = new_situation
		if(new_marketplace is not None):
			self.marketplace = new_marketplace
		if(new_agents is not None):
			self.agents = new_agents
			color_map = self.get_cmap(len(self.agents))
			self.agent_colors = []
			for i in range(0, len(self.agents)):
				self.agent_colors.append(color_map(i))

	# metrics
	def metrics_average(self, rewards) -> np.float64:
		return np.mean(np.array(rewards))

	def metrics_median(self, rewards) -> np.float64:
		return np.median(np.array(rewards))

	def metrics_maximum(self, rewards) -> np.float64:
		return np.max(np.array(rewards))

	def metrics_minimum(self, rewards) -> np.float64:
		return np.min(np.array(rewards))

	# visualize metrics
	def create_histogram(self, rewards) -> None:
		plt.xlabel('Reward', fontsize='18')
		plt.ylabel('Episodes', fontsize='18')
		plt.hist(rewards, bins=10, align='mid', color=self.agent_colors, edgecolor='black', range=(0, self.round_up(int(self.metrics_maximum(rewards)), -3)))
		plt.legend(self.agents)
		if self.enable_live_draws:
			plt.draw()
			plt.pause(0.001)

	def run_marketplace(self) -> list:
		# initialize the rewards list with a list for each agent
		rewards = []
		for i in range(len(self.agents)):
			rewards.append([])

		for episode in range(1, self.episodes + 1):
			# reset the state once to be used by all agents
			default_state = self.marketplace.reset()

			for i in range(0, len(self.agents)):
				# reset values for all agents
				state = default_state
				reward_per_episode = 0
				is_done = False

				# run marketplace for this agent
				while not is_done:
					action = self.agents[i].policy(state)
					state, reward, is_done, _ = self.marketplace.step(action)
					reward_per_episode += reward

				# add the reward to the current agent's reward-Array
				rewards[i].append(reward_per_episode)

			# after all agents have run the episode
			if (episode % 100) == 0:
				print(f'Running {episode}th episode...')

			if (episode % self.histogram_plot_interval) == 0:
				self.create_histogram(rewards)

		return rewards


monitor = Monitor()


def main():
	import agent
	monitor.setup_monitoring(new_agents=[monitor.agents[0], agent.FixedPriceAgent(6), agent.FixedPriceAgent(2)])
	print(f'Running', monitor.episodes, 'episodes')
	print(f'Plot interval is:', monitor.histogram_plot_interval)
	print(f'Using modelfile: ' + monitor.path_to_modelfile)
	print(f'The situation is: ' + monitor.situation)
	print(f'The marketplace is:', monitor.marketplace)
	print(f'Monitoring these agents:')
	for current_agent in monitor.agents:
		print(current_agent)

	rewards = monitor.run_marketplace()

	for i in range(len(rewards)):
		print(f'Statistics for agent:', monitor.agents[i])
		print(f'The average reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_average(rewards[i])))
		print(f'The median reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_median(rewards[i])))
		print(f'The maximum reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_maximum(rewards[i])))
		print(f'The minimum reward over {monitor.episodes} episodes is: ' + str(monitor.metrics_minimum(rewards[i])))

	# save last histogram
	fname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + os.sep + 'monitoring' + os.sep + 'final_plot_' + time.strftime('%Y%m%d-%H%M%S') + '.svg'
	plt.savefig(fname=fname)
	print(f'The final histogram was saved at: ' + fname)


if __name__ == '__main__':
	main()
