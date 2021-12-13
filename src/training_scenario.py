import torch

import agent
import sim_market as sim
import training

economy = sim.CircularEconomy()
n_actions = 1
for id in range(0, len(economy.action_space)):
	n_actions *= economy.action_space[id].n

RL_agent = agent.QLearningCEAgent(economy.observation_space.shape[0], n_actions, optim=torch.optim.Adam)
training.train_QLearning_agent(RL_agent, economy)
