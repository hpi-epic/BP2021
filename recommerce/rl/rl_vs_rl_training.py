import os

import matplotlib.pyplot as plt
import numpy as np
from attrdict import AttrDict

from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC


def train_rl_vs_rl(
		config_market: AttrDict,
		config_rl1: AttrDict,
		config_rl2: AttrDict,
		num_switches: int = 30,
		num_steps_per_switch: int = 25000):
	tmp_marketplace = CircularEconomyRebuyPriceDuopoly(config=config_market)
	agent1 = StableBaselinesPPO(config_market=config_market, config_rl=config_rl1, marketplace=tmp_marketplace)
	marketplace_for_agent2 = CircularEconomyRebuyPriceDuopoly(config=config_market, competitors=[agent1])
	agent2 = StableBaselinesSAC(config_market=config_market, config_rl=config_rl2, marketplace=marketplace_for_agent2)
	marketplace_for_agent1 = CircularEconomyRebuyPriceDuopoly(config=config_market, competitors=[agent2])
	agent1.set_marketplace(marketplace_for_agent1)
	agents = [agent1, agent2]
	assert len(agents) == 2, 'This scenario is only for exactly two agents.'

	rewards = [[], []]

	for i in range(num_switches):
		print(f'\n\nTraining {i + 1}\nTraining generation {i // 2 + 1} of {agents[i % 2].name}.')
		last_dicts = agents[i % 2].train_agent(training_steps=num_steps_per_switch).all_dicts
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
	plt.savefig(os.path.join(PathManager.results_path, 'monitoring', 'rl_vs_rl_all.svg'), transparent=True)  # Maybe save in a dedicated folder
