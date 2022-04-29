import copy

import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.q_learning.q_learning_training as q_learning_training
from recommerce.configuration.hyperparameter_config import HyperparameterConfig, HyperparameterConfigLoader
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

config_hyperparameter: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')


test_scenarios = [
	linear_market.LinearEconomyDuopoly,
	linear_market.LinearEconomyOligopoly,
	circular_market.CircularEconomyMonopoly,
	circular_market.CircularEconomyRebuyPriceMonopoly,
	circular_market.CircularEconomyRebuyPriceDuopoly
]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('marketplace_class', test_scenarios)
def test_market_scenario(marketplace_class):
	mock_config = copy.deepcopy(config_hyperparameter)
	mock_config.replay_start_size = 500
	mock_config.sync_target_frames = 100
	q_learning_training.QLearningTrainer(
		marketplace_class=marketplace_class,
		agent_class=QLearningAgent,
		config=mock_config
		).train_agent(int(mock_config.replay_start_size * 1.2))


@pytest.mark.training
@pytest.mark.slow
def test_training_with_tensorboard():
	mock_config = copy.deepcopy(config_hyperparameter)
	mock_config.replay_start_size = 500
	mock_config.sync_target_frames = 100

	marketplace_class = linear_market.LinearEconomyDuopoly
	agent_class = QLearningAgent

	q_learning_training.QLearningTrainer(
		marketplace_class=marketplace_class,
		agent_class=agent_class,
		config=mock_config
		).train_agent(int(mock_config.replay_start_size * 1.2))
