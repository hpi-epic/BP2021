import os
import re
from unittest.mock import patch

import agents.vendors as vendors
import monitoring.agent_monitoring.am_monitoring as monitoring

monitor = monitoring.Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(enable_live_draw=False, subfolder_name=f'test_plots_{function.__name__}')


def teardown_module(module):
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'results', 'monitoring')):
		if re.match('test_*', file_name):
			assert False, 'file writing was not mocked or a created file was not removed after the test!'


def test_run_marketplace():
	monitor.configurator.setup_monitoring(episodes=100, plot_interval=100, agents=[(vendors.FixedPriceCEAgent, [(5, 2)])])
	with patch('monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		agent_rewards = monitor.run_marketplace()
		assert 1 == len(agent_rewards)
		assert monitor.configurator.episodes == len(agent_rewards[0])


def test_run_monitoring_session():
	monitor.configurator.setup_monitoring(episodes=10, plot_interval=10)
	current_configuration = monitor.configurator.get_configuration()
	with patch('monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		monitoring.run_monitoring_session(monitor)
		assert current_configuration == monitor.configurator.get_configuration(), 'the monitor configuration should not be changed within run_monitoring()'
		assert os.path.exists(monitor.configurator.folder_path)


def test_run_monitoring_ratio():
	# ratio is over 50, program should ask if we want to continue. We answer 'no'
	with patch('monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('monitoring.agent_monitoring.am_configuration.input', create=True) as mocked_input, \
		patch('monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		mocked_input.side_effect = ['n']
		exists_mock.return_value = True
		monitor.configurator.setup_monitoring(episodes=51, plot_interval=1)
		monitoring.run_monitoring_session(monitor)
