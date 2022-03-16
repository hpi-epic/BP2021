import os
import re
from unittest.mock import patch

import alpha_business.agents.vendors as vendors
import alpha_business.market.circular.circular_sim_market as circular_market
import alpha_business.monitoring.agent_monitoring.am_monitoring as monitoring
from alpha_business.configuration.path_manager import PathManager

monitor = monitoring.Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(
		enable_live_draw=False,
		episodes=50,
		plot_interval=10,
		marketplace=circular_market.CircularEconomyMonopolyScenario,
		agents=[(vendors.QLearningCEAgent, [os.path.join(os.path.dirname(__file__), os.pardir, 'test_data',
			'CircularEconomyMonopolyScenario_QLearningCEAgent.dat')])],
		subfolder_name=f'test_plots_{function.__name__}')


def teardown_module(module):
	for file_name in os.listdir(os.path.join(PathManager.results_path, 'monitoring')):
		if re.match('test_*', file_name):
			assert False, 'Test files were not mocked correctly'


def test_run_marketplace():
	monitor.configurator.setup_monitoring(episodes=100, plot_interval=100, agents=[(vendors.FixedPriceCEAgent, [(5, 2)])])
	with patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		agent_rewards = monitor.run_marketplace()
		assert 1 == len(agent_rewards)
		assert monitor.configurator.episodes == len(agent_rewards[0])


def test_run_monitoring_session():
	monitor.configurator.setup_monitoring(episodes=10, plot_interval=10)
	current_configuration = monitor.configurator.get_configuration()
	with patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		monitoring.run_monitoring_session(monitor)
		assert current_configuration == monitor.configurator.get_configuration(), \
			'the monitor configuration should not be changed within run_monitoring()'
		assert os.path.exists(monitor.configurator.folder_path)


def test_run_monitoring_ratio():
	# ratio is over 50, program should ask if we want to continue. We answer 'no'
	with patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.input', create=True) as mocked_input, \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		mocked_input.side_effect = ['n']
		exists_mock.return_value = True
		monitor.configurator.setup_monitoring(episodes=51, plot_interval=1)
		monitoring.run_monitoring_session(monitor)
