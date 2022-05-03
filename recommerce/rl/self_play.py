import matplotlib.pyplot as plt
import numpy as np

from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly, CircularEconomyRebuyPriceVariableDuopoly
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent, StableBaselinesPPO


def train_self_play(agent_class: StableBaselinesAgent = StableBaselinesPPO, training_steps=1000000):
	tmp_marketplace = CircularEconomyRebuyPriceDuopoly(True)
	agent = agent_class(tmp_marketplace)
	marketplace = CircularEconomyRebuyPriceVariableDuopoly(agent)
	agent.set_marketplace(marketplace)

	rewards = [[], []]
	last_dicts = agent.train_agent(training_steps=training_steps, analyze_after_training=False)
	for mydict in last_dicts:
		rewards[0].append(mydict['profits/all']['vendor_0'])
		rewards[1].append(mydict['profits/all']['vendor_1'])

	# This is a bit hacky. It should be replaced by better monitoring later.
	smoothed_return_estimation = \
		[[np.mean(rewards[idx][max(i-50, 0):i]) for i in range(len(rewards[idx]))] for idx in range(2)]
	print(rewards)
	print('\n\n')
	print(smoothed_return_estimation)
	plt.clf()
	plt.plot(smoothed_return_estimation[0], label='policy under training')
	plt.plot(smoothed_return_estimation[1], label='same policy (opponent)')
	plt.legend()
	# plt.show()  # Save the figure later in a nice folder.
