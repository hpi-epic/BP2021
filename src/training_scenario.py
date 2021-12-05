import torch

import agent
import sim_market as sim
import training

eco = sim.CircularEconomyRebuyPrice()
RL_agent = agent.QLearningCERebuy(eco.observation_space.shape[0], 1000, optim=torch.optim.Adam)
training.train_QLearning_agent(RL_agent, eco)
