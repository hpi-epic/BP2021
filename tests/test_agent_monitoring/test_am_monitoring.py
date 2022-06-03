import os
from unittest.mock import patch

import pytest
from attrdict import AttrDict

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.monitoring.agent_monitoring.am_monitoring as monitoring
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_vendors import FixedPriceCEAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

monitor = monitoring.Monitor()

config_market: AttrDict = HyperparameterConfigLoader.load('market_config')


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(
		enable_live_draw=False,
		episodes=50,
		plot_interval=10,
		marketplace=circular_market.CircularEconomyMonopoly,
		agents=[(QLearningAgent, [os.path.join(os.path.dirname(__file__), os.pardir, 'test_data',
			'CircularEconomyMonopoly_QLearningAgent.dat')])],
		config_market=config_market)


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
		analysis_results = monitor.run_marketplace()
		assert 1 == len(analysis_results)
		assert monitor.configurator.episodes == len(ut.unroll_dict_with_list(analysis_results[0])['profits/all/vendor_0'])


def test_run_monitoring_session():
	monitor.configurator.setup_monitoring(episodes=10, plot_interval=10, config_market=config_market)
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
		monitor.configurator.setup_monitoring(episodes=51, plot_interval=1, config_market=config_market)
		monitoring.run_monitoring_session(monitor)
