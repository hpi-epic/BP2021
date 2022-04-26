from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceOneCompetitor, CircularEconomyVariableDuopoly
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesPPO, StableBaselinesSAC

# import numpy as np


def train_rl_vs_rl():
    old_marketplace = CircularEconomyRebuyPriceOneCompetitor(True)
    agent1 = StableBaselinesPPO(old_marketplace)
    marketplace_for_agent2 = CircularEconomyVariableDuopoly(agent1)
    agent2 = StableBaselinesSAC(marketplace_for_agent2)
    marketplace_for_agent1 = CircularEconomyVariableDuopoly(agent2)
    agent1.model.set_env(marketplace_for_agent1)
    agents = [agent1, agent2]

    rewards = [[], []]

    for i in range(10):
        print(f'\n\nTraining {i}\nNow I train the {i // 2}th generation of {agents[i % 2].name}.')
        last_dicts = agents[i % 2].train_agent(training_steps=10000, analyze_after_training=False)
        for mydict in last_dicts:
            rewards[i % 2].append(mydict['profits/all']['vendor_0'])
            rewards[(i + 1) % 2].append(mydict['profits/all']['vendor_1'])

    # return_estimation = [[np.sum(rewards[agent_idx][(i * 50):(i + 1) * 50])] for agent_idx in range(2)]


if __name__ == '__main__':
    train_rl_vs_rl()
