import os
import re
import shutil
from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import torch

from .context import agent, sim_market, training
from .context import utils_rl as ut_rl


# teardown after each test
def teardown_module(module):
	print('***TEARDOWN***')
	for f in os.listdir('./runs'):
		if re.match('test_*', f):
			shutil.rmtree('./runs/' + f)


# Helper function that returns a mock config_rl.json file/string with the given values
def create_mock_json(gamma='0.99', batch_size='32', replay_size='100000', learning_rate='1e-6', sync_target_frames='1000', replay_start_size='10000', epsilon_decay_last_frame='75000', epsilon_start='1.0', epsilon_final='0.1'):
	return '{\n\t"gamma" : ' + gamma + ',\n' + \
		'\t"batch_size" : ' + batch_size + ',\n' + \
		'\t"replay_size" : ' + replay_size + ',\n' + \
		'\t"learning_rate" : ' + learning_rate + ',\n' + \
		'\t"sync_target_frames" : ' + sync_target_frames + ',\n' + \
		'\t"replay_start_size" : ' + replay_start_size + ',\n' + \
		'\t"epsilon_decay_last_frame" : ' + epsilon_decay_last_frame + ',\n' + \
		'\t"epsilon_start" : ' + epsilon_start + ',\n' + \
		'\t"epsilon_final" : ' + epsilon_final + '\n' + \
		'}'


# Helper function to test if the mock_file is setup correctly
def check_mock_file(mock_file, json=create_mock_json()):
	path = os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config_rl.json'
	assert (open(path).read() == json)
	mock_file.assert_called_with(path)
	ut_rl.config = ut_rl.load_config(path)


test_scenarios = [
	(sim_market.ClassicScenario(), agent.QLearningAgent(n_observation=sim_market.ClassicScenario().observation_space.shape[0], n_actions=10, optim=torch.optim.Adam)),
	(sim_market.MultiCompetitorScenario(), agent.QLearningAgent(n_observation=sim_market.MultiCompetitorScenario().observation_space.shape[0], n_actions=10, optim=torch.optim.Adam)),
	(sim_market.CircularEconomy(), agent.QLearningCEAgent(sim_market.CircularEconomy().observation_space.shape[0], n_actions=100, optim=torch.optim.Adam)),
	(sim_market.CircularEconomyRebuyPrice(), agent.QLearningCERebuyAgent(sim_market.CircularEconomyRebuyPrice().observation_space.shape[0], n_actions=100, optim=torch.optim.Adam))]


@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_market_scenario(environment, agent):
	json = create_mock_json(replay_start_size='500')
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		check_mock_file(mock_file, json)
		# Include utils_rl again to make sure the file is read again
		reload(ut_rl)
		training.train_QLearning_agent(agent, environment, int(ut_rl.REPLAY_START_SIZE * 1.2), log_dir_prepend='test_')
