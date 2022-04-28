import json
from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import recommerce.configuration.hyperparameter_config as hyperparameter_config
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.q_learning.q_learning_training as q_learning_training
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader, HyperparameterConfig

config_hyperparameter: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')

def teardown_module(module):
	print('***TEARDOWN***')
	reload(hyperparameter_config)


def import_config() -> hyperparameter_config.HyperparameterConfig:
	"""
	Reload the hyperparameter_config file to update the config variable with the mocked values.

	Returns:
		HyperparameterConfig: The config object.
	"""
	reload(hyperparameter_config)
	return hyperparameter_config.config


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
	mock_config = config_hyperparameter
	mock_config.replay_start_size = 500
	mock_config.sync_target_frames = 100
	q_learning_training.QLearningTrainer(marketplace_class=marketplace_class, agent_class=QLearningAgent, config=mock_config).train_agent(int(mock_config.replay_start_size * 1.2))


@pytest.mark.training
@pytest.mark.slow
def test_training_with_tensorboard():
	mock_config = config_hyperparameter
	mock_config.replay_start_size = 500
	mock_config.sync_target_frames = 100
	
	marketplace_class = linear_market.LinearEconomyDuopoly
	agent_class = QLearningAgent

	q_learning_training.QLearningTrainer(marketplace_class=marketplace_class, agent_class=agent_class, config=mock_config).train_agent(int(mock_config.replay_start_size * 1.2))
