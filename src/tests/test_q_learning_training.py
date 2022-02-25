import os
import re
import shutil
import time
from importlib import reload
from unittest.mock import mock_open, patch

import pytest

import agents.vendors as vendors
import configuration.hyperparameters_config as config
import market.circular.circular_sim_market as circular_market
import market.linear.linear_sim_market as linear_market
import rl.q_learning_training as q_learning_training
import tests.utils_tests as ut_t


def teardown_module(module):
	print('***TEARDOWN***')
	# we need to sleep because sometimes the subfolder is still being used when we try to remove it
	time.sleep(0.001)
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'trainedModels')):
		if re.match('test_*', file_name):
			shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'trainedModels', file_name))
	reload(config)


test_scenarios = [
	(linear_market.ClassicScenario, vendors.QLearningLEAgent),
	(linear_market.MultiCompetitorScenario, vendors.QLearningLEAgent),
	(circular_market.CircularEconomyMonopolyScenario, vendors.QLearningCEAgent),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, vendors.QLearningCERebuyAgent),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, vendors.QLearningCERebuyAgent)
]


@pytest.mark.parametrize('market_class, agent_class', test_scenarios)
def test_market_scenario(market_class, agent_class):
	json = ut_t.create_hyperparameter_mock_json(rl=ut_t.create_hyperparameter_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file, \
		patch('rl.training.SummaryWriter'), \
		patch('agents.vendors.QLearningAgent.save'):
		ut_t.check_mock_file(mock_file, json)
		# Include config again to make sure the file is read again
		reload(config)
		q_learning_training.QLearningTrainer(market_class, agent_class, log_dir_prepend='test_').train_agent(int(config.REPLAY_START_SIZE * 1.2))


def test_training_with_tensorboard():
	json = ut_t.create_hyperparameter_mock_json(rl=ut_t.create_hyperparameter_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file, \
		patch('rl.training.SummaryWriter'), \
		patch('agents.vendors.QLearningAgent.save'):
		ut_t.check_mock_file(mock_file, json)
		# Include utils_rl again to make sure the file is read again
		reload(config)
		market_class = linear_market.ClassicScenario
		agent_class = vendors.QLearningLEAgent
		q_learning_training.QLearningTrainer(market_class, agent_class, log_dir_prepend='test_').train_agent(int(config.REPLAY_START_SIZE * 1.2))
