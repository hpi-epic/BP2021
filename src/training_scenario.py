import torch

import sim_market as sim
import training
import vendors

economy = sim.CircularEconomyRebuyPriceOneCompetitor()
n_actions = 1
for id in range(0, len(economy.action_space)):
	n_actions *= economy.action_space[id].n

RL_agent = vendors.QLearningCERebuyAgent(n_observation=economy.observation_space.shape[0], n_actions=n_actions, optim=torch.optim.Adam)
training.train_QLearning_agent(RL_agent, economy)
