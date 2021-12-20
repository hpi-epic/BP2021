import pytest
import torch

from .context import agent, sim_market, training
from .context import utils_rl as ut_rl

test_scenarios = [
    (sim_market.ClassicScenario(), agent.QLearningAgent(n_observation=sim_market.ClassicScenario().observation_space.shape[0], n_actions=10, optim=torch.optim.Adam)),
    (sim_market.MultiCompetitorScenario(), agent.QLearningAgent(n_observation=sim_market.MultiCompetitorScenario().observation_space.shape[0], n_actions=10, optim=torch.optim.Adam)),
    (sim_market.CircularEconomy(), agent.QLearningCEAgent(sim_market.CircularEconomy().observation_space.shape[0], n_actions=100, optim=torch.optim.Adam)),
    (sim_market.CircularEconomyRebuyPrice(), agent.QLearningCERebuyAgent(sim_market.CircularEconomyRebuyPrice().observation_space.shape[0], n_actions=100, optim=torch.optim.Adam))]


@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_market_scenario(environment, agent):
    training.train_QLearning_agent(agent, environment, int(ut_rl.REPLAY_START_SIZE * 1.2))
