import json
import os
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.monitoring.agent_monitoring.am_monitoring as monitoring
from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.market.circular.circular_vendors import FixedPriceCEAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

monitor = monitoring.Monitor()

config_hyperparameter: HyperparameterConfig = None


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor, config_hyperparameter
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(
		rl=ut_t.create_hyperparameter_mock_dict_rl(
			replay_size=500, sync_target_frames=10, replay_start_size=100, epsilon_decay_last_frame=400),
		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(
			max_storage=100, episode_length=25, max_price=10, max_quality=50,
			number_of_customers=10, production_price=3, storage_cost_per_product=0.1
		)
	))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		config_hyperparameter = ut_t.import_config()
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(
		enable_live_draw=False,
		episodes=50,
		plot_interval=10,
		marketplace=circular_market.CircularEconomyMonopoly,
		agents=[(QLearningAgent, [os.path.join(os.path.dirname(__file__), os.pardir, 'test_data',
			'CircularEconomyMonopoly_QLearningAgent.dat')])],
		config=config_hyperparameter,
		subfolder_name=f'test_plots_{function.__name__}')


def test_run_marketplace():
	monitor.configurator.setup_monitoring(
		episodes=100,
		plot_interval=100,
		agents=[(FixedPriceCEAgent, [(5, 2)])]
		)
	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		agent_rewards = monitor.run_marketplace()
		assert 1 == len(agent_rewards)
		assert monitor.configurator.episodes == len(agent_rewards[0])


def test_run_monitoring_session():
	monitor.configurator.setup_monitoring(episodes=10, plot_interval=10, config=config_hyperparameter)
	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		monitoring.run_monitoring_session(monitor)
		assert os.path.exists(monitor.configurator.folder_path)


@pytest.mark.slow
def test_run_monitoring_ratio():
	# ratio is over 50, program should ask if we want to continue. We answer 'no'
	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.input', create=True) as mocked_input, \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		mocked_input.side_effect = ['n']
		exists_mock.return_value = True
		monitor.configurator.setup_monitoring(episodes=51, plot_interval=1, config=config_hyperparameter)
		monitoring.run_monitoring_session(monitor)
