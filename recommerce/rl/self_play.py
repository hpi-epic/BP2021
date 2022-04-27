import matplotlib.pyplot as plt
import numpy as np

from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly, CircularEconomyVariableDuopoly
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesPPO  # , StableBaselinesSAC


def train_self_play():
	old_marketplace = CircularEconomyRebuyPriceDuopoly(True)
	agent = StableBaselinesPPO(old_marketplace)
	marketplace = CircularEconomyVariableDuopoly(agent)
	agent.model.set_env(marketplace)

	rewards = [[], []]
	last_dicts = agent.train_agent(training_steps=200000, analyze_after_training=False)
	for mydict in last_dicts:
		rewards[0].append(mydict['profits/all']['vendor_0'])
		rewards[1].append(mydict['profits/all']['vendor_1'])

	# This is a bit hacky. It should be replaced by better monitoring later.
	return_estimation = [[np.sum(rewards[idx][(i * 50):(i + 1) * 50]) for i in range(len(rewards[idx]) // 50)] for idx in range(2)]
	smoothed_return_estimation = \
		[[np.mean(return_estimation[idx][max(i-50, 0):i]) for i in range(len(return_estimation[idx]))] for idx in range(2)]
	print(return_estimation)
	print('\n\n')
	print(smoothed_return_estimation)
	plt.plot(smoothed_return_estimation[0], label='version 1')
	plt.plot(smoothed_return_estimation[1], label='version 2')
	plt.legend()
	plt.show()  # Save the figure later in a nice folder.


if __name__ == '__main__':
	train_self_play()
