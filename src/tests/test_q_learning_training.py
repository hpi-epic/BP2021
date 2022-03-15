import os
import re
import shutil
import time
from importlib import reload
from unittest.mock import mock_open, patch

import pytest

import configuration.hyperparameter_config as hyperparameter_config
import market.circular.circular_sim_market as circular_market
import market.linear.linear_sim_market as linear_market
import rl.q_learning_training as q_learning_training
import tests.utils_tests as ut_t
from rl.q_learning_agent import QLearningCEAgent, QLearningCERebuyAgent, QLearningLEAgent


def teardown_module(module):
	print('***TEARDOWN***')
	# we need to sleep because sometimes the subfolder is still being used when we try to remove it
	time.sleep(0.001)
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'trainedModels')):
		if re.match('test_*', file_name):
			shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'trainedModels', file_name))
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
	(linear_market.ClassicScenario, QLearningLEAgent),
	(linear_market.MultiCompetitorScenario, QLearningLEAgent),
	(circular_market.CircularEconomyMonopolyScenario, QLearningCEAgent),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, QLearningCERebuyAgent),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, QLearningCERebuyAgent)
]


@pytest.mark.parametrize('market_class, agent_class', test_scenarios)
def test_market_scenario(market_class, agent_class):
	json = ut_t.create_hyperparameter_mock_json(rl=ut_t.create_hyperparameter_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file, \
		patch('rl.training.SummaryWriter'), \
		patch('rl.q_learning_agent.QLearningAgent.save'):
		ut_t.check_mock_file(mock_file, json)

		config = import_config()
		q_learning_training.QLearningTrainer(market_class, agent_class, log_dir_prepend='test_').train_agent(int(config.replay_start_size * 1.2))


def test_training_with_tensorboard():
	json = ut_t.create_hyperparameter_mock_json(rl=ut_t.create_hyperparameter_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file, \
		patch('rl.training.SummaryWriter'), \
		patch('rl.q_learning_agent.QLearningAgent.save'):
		ut_t.check_mock_file(mock_file, json)

		config = import_config()
		market_class = linear_market.ClassicScenario
		agent_class = QLearningLEAgent
		q_learning_training.QLearningTrainer(market_class, agent_class, log_dir_prepend='test_').train_agent(int(config.replay_start_size * 1.2))
