import torch

import agent
import sim_market as sim
import training

economy = sim.ClassicScenario()
RL_agent = agent.QLearningAgent(economy.observation_space.shape[0], economy.action_space.n, optim=torch.optim.Adam)
training.train_QLearning_agent(RL_agent, economy)
