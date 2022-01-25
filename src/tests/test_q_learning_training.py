import os
import re
import shutil
import time
from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import torch

import agents.vendors as vendors
import configuration.config as config
import market.circular.circular_sim_market as circular_market
import market.linear.linear_sim_market as linear_market
import rl.q_learning_training as q_learning_training
import tests.utils_tests as ut_t


def teardown_module(module):
	reload(config)


test_scenarios = [
	(linear_market.ClassicScenario, vendors.QLearningAgent),
	(linear_market.MultiCompetitorScenario, vendors.QLearningAgent),
	(circular_market.CircularEconomyMonopolyScenario, vendors.QLearningCEAgent),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, vendors.QLearningCERebuyAgent),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, vendors.QLearningCERebuyAgent)
]


@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_market_scenario(environment, agent):
	json = ut_t.create_mock_json(rl=ut_t.create_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file, \
		patch('rl.training.SummaryWriter'):
		ut_t.check_mock_file(mock_file, json)
		# Include config again to make sure the file is read again
		reload(config)
		environment = environment()
		agent = agent(environment.observation_space.shape[0], n_actions=environment.get_n_actions(), optim=torch.optim.Adam)
		q_learning_training.QLearningTrainer(environment, agent).train_agent(int(config.REPLAY_START_SIZE * 1.2))


def test_training_with_tensorboard():
	json = ut_t.create_mock_json(rl=ut_t.create_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		# Include utils_rl again to make sure the file is read again
		reload(config)
		environment = linear_market.ClassicScenario()
		agent = vendors.QLearningAgent(environment.observation_space.shape[0], n_actions=environment.get_n_actions(), optim=torch.optim.Adam)
		q_learning_training.QLearningTrainer(environment, agent, log_dir_prepend='test_').train_agent(int(config.REPLAY_START_SIZE * 1.2))

	print('***TEARDOWN***')
	# we need to sleep because sometimes the runs folder is still being used when we try to remove it
	time.sleep(0.001)
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs')):
		if re.match('test_*', file_name):
			shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs', file_name))
	# remove the runs folder if it is empty, because that means it has only been created for our tests
	if os.listdir(os.path.join('results', 'runs')) == []:
		os.rmdir(os.path.join('results', 'runs'))
