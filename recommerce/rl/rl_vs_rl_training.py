import os

import matplotlib.pyplot as plt
import numpy as np

from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly, CircularEconomyRebuyPriceVariableDuopoly
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesPPO, StableBaselinesSAC


def train_rl_vs_rl(num_switches: int = 30, num_steps_per_switch: int = 25000):
	tmp_marketplace = CircularEconomyRebuyPriceDuopoly(True)
	agent1 = StableBaselinesPPO(tmp_marketplace)
	marketplace_for_agent2 = CircularEconomyRebuyPriceVariableDuopoly(agent1)
	agent2 = StableBaselinesSAC(marketplace_for_agent2)
	marketplace_for_agent1 = CircularEconomyRebuyPriceVariableDuopoly(agent2)
	agent1.set_marketplace(marketplace_for_agent1)
	agents = [agent1, agent2]
	assert len(agents) == 2, 'This scenario is only for exactly two agents.'

	rewards = [[], []]

	for i in range(num_switches):
		print(f'\n\nTraining {i + 1}\nTraining generation {i // 2 + 1} of {agents[i % 2].name}.')
		last_dicts = agents[i % 2].train_agent(training_steps=num_steps_per_switch)
		for mydict in last_dicts:
			rewards[i % 2].append(mydict['profits/all']['vendor_0'])
			rewards[(i + 1) % 2].append(mydict['profits/all']['vendor_1'])

	# This is a bit hacky. But it isn't used so often, so I will leave it for now.
	# More heavy changes come with this.
	smoothed_return_estimation = \
		[[np.mean(rewards[idx][max(i-50, 0):(i + 1)]) for i in range(len(rewards[idx]))] for idx in range(2)]
	plt.clf()
	plt.plot(smoothed_return_estimation[0], label=agents[0].name)
	plt.plot(smoothed_return_estimation[1], label=agents[1].name)
	plt.legend()
	plt.savefig(os.path.join(PathManager.results_path, 'monitoring', 'rl_vs_rl_all.svg'))  # Maybe save in a dedicated folder
