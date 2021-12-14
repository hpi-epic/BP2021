import torch

import agent
import sim_market as sim
import training

economy = sim.CircularEconomyRebuyPrice()

n_actions = 1
for id in range(0, len(economy.action_space)):
	n_actions *= economy.action_space[id].n

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'

RL_agent = agent.QLearningCERebuyAgent(n_observation=economy.observation_space.shape[0], n_actions=n_actions, device=device, optim=torch.optim.Adam)
training.train_QLearning_agent(RL_agent, economy)
