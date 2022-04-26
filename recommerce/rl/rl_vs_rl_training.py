import matplotlib.pyplot as plt
import numpy as np

from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceOneCompetitor, CircularEconomyVariableDuopoly
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesPPO, StableBaselinesSAC


def train_rl_vs_rl():
    old_marketplace = CircularEconomyRebuyPriceOneCompetitor(True)
    agent1 = StableBaselinesPPO(old_marketplace)
    marketplace_for_agent2 = CircularEconomyVariableDuopoly(agent1)
    agent2 = StableBaselinesSAC(marketplace_for_agent2)
    marketplace_for_agent1 = CircularEconomyVariableDuopoly(agent2)
    agent1.model.set_env(marketplace_for_agent1)
    agents = [agent1, agent2]

    rewards = [[], []]

    for i in range(20):
        print(f'\n\nTraining {i + 1}\nNow I train generaion {i // 2 + 1} of {agents[i % 2].name}.')
        last_dicts = agents[i % 2].train_agent(training_steps=50000, analyze_after_training=False)
        for mydict in last_dicts:
            rewards[i % 2].append(mydict['profits/all']['vendor_0'])
            rewards[(i + 1) % 2].append(mydict['profits/all']['vendor_1'])

    return_estimation = [[np.sum(rewards[idx][(i * 50):(i + 1) * 50]) for i in range(len(rewards[idx]) // 50)] for idx in range(2)]
    smoothed_return_estimation = \
        [[np.mean(return_estimation[idx][max(i-50, 0):i]) for i in range(len(return_estimation[idx]))] for idx in range(2)]
    plt.plot(smoothed_return_estimation[0], label=agents[0].name)
    plt.plot(smoothed_return_estimation[1], label=agents[1].name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_rl_vs_rl()
