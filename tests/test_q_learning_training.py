from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import recommerce.configuration.hyperparameter_config as hyperparameter_config
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.q_learning.q_learning_training as q_learning_training
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent


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
	(linear_market.ClassicScenario, QLearningAgent),
	(linear_market.MultiCompetitorScenario, QLearningAgent),
	(circular_market.CircularEconomyMonopolyScenario, QLearningAgent),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, QLearningAgent),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, QLearningAgent)
]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('market_class, agent_class', test_scenarios)
def test_market_scenario(market_class, agent_class):
	json = ut_t.create_hyperparameter_mock_json(rl=ut_t.create_hyperparameter_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		config = import_config()

	q_learning_training.QLearningTrainer(market_class, agent_class).train_agent(int(config.replay_start_size * 1.2))


@pytest.mark.training
@pytest.mark.slow
def test_training_with_tensorboard():
	json = ut_t.create_hyperparameter_mock_json(rl=ut_t.create_hyperparameter_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		config = import_config()
		market_class = linear_market.ClassicScenario
		agent_class = QLearningAgent

	q_learning_training.QLearningTrainer(market_class, agent_class).train_agent(int(config.replay_start_size * 1.2))
