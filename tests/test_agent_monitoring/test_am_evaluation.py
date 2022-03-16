import os
import re
from unittest.mock import patch

import numpy as np
import pytest

import alpha_business.market.circular.circular_sim_market as circular_market
import alpha_business.monitoring.agent_monitoring.am_monitoring as monitoring
from alpha_business.configuration.path_manager import PathManager
from alpha_business.market.circular.circular_vendors import FixedPriceCEAgent, RuleBasedCEAgent
from alpha_business.rl.q_learning.q_learning_agent import QLearningCEAgent

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
		agents=[(QLearningCEAgent, [os.path.join(os.path.dirname(__file__), os.pardir, 'test_data',
			'CircularEconomyMonopolyScenario_QLearningCEAgent.dat')])],
		subfolder_name=f'test_plots_{function.__name__}')


def teardown_module(module):
	for file_name in os.listdir(os.path.join(PathManager.results_path, 'monitoring')):
		if re.match('test_*', file_name):
			assert False, 'Test files were not mocked correctly'


evaluate_session_testcases = [
	([(RuleBasedCEAgent, [])], [[5, 10, 0, -5]]),
	([(RuleBasedCEAgent, []), (FixedPriceCEAgent, [])], [[5, 10, 0, -5], [5, -10, 60, 5]])
]


@pytest.mark.parametrize('agents, rewards', evaluate_session_testcases)
def test_evaluate_session(agents, rewards):
	with patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.clf'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.xlabel'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.title'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.legend'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.pause'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.draw'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.savefig'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		monitor.configurator.setup_monitoring(episodes=4, plot_interval=1, agents=agents)
		monitor.evaluator.evaluate_session(rewards)


# all arrays in rewards must be of the same size
def test_rewards_array_size():
	# Numpy doesn't like nested arrays of different sizes, need to specify dtype=object
	rewards_wrong = np.array([[1, 2], [1, 2, 3]], dtype=object)

	with patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt'):
		with pytest.raises(AssertionError) as assertion_message:
			monitor.evaluator.create_histogram(rewards_wrong)
		assert 'all rewards-arrays must be of the same size' in str(assertion_message.value)


create_histogram_statistics_plots_testcases = [
	([(RuleBasedCEAgent, [])], [[100, 0]], 1, [(1.0, 0.0, 0.0, 1.0)], (0.0, 1000.0)),
	([(RuleBasedCEAgent, []), (RuleBasedCEAgent, [])],
		[[100, 0], [10, 5]], 1, [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0)], (0.0, 1000.0)),
	([(RuleBasedCEAgent, []), (RuleBasedCEAgent, []), (RuleBasedCEAgent, []), (RuleBasedCEAgent, [])],
		[[100, 0], [10, 5], [100, 10000], [10, 1000]],
		10,
		[(1.0, 0.0, 0.0, 1.0), (0.5234360234360234, 1.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0), (0.4296860234360234, 0.0, 1.0, 1.0)],
		(0.0, 10000.0))
]


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', create_histogram_statistics_plots_testcases)
def test_create_histogram(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.configurator.setup_monitoring(enable_live_draw=True, agents=agents)
	name_list = [agent.name for agent in monitor.configurator.agents]
	with patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.clf'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.xlabel'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.title'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.hist') as hist_mock, \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.legend') as legend_mock, \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.pause'), \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.draw') as draw_mock, \
		patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt.savefig') as save_mock, \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.evaluator.create_histogram(rewards)
		hist_mock.assert_called_once_with(rewards, bins=plot_bins, color=agent_color, rwidth=0.9, range=lower_upper_range, edgecolor='black')
		legend_mock.assert_called_once_with(name_list)
		draw_mock.assert_called_once()
		save_mock.assert_called_once_with(fname=os.path.join(monitor.configurator.folder_path, 'histograms', 'default.svg'))


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', create_histogram_statistics_plots_testcases)
def test_create_statistics_plots(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.configurator.setup_monitoring(agents=agents, episodes=len(rewards[0]), plot_interval=1)
	with patch('alpha_business.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('alpha_business.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.evaluator._create_statistics_plots(rewards)


incorrect_create_line_plot_testcases = [
	([1, 2, 3], [[2], [1]], 'Overall', 'x_values must have self.episodes / self.plot_interval many items'),
	([1, 2], [[2], [1]], 'Overall', 'y_values must have one entry per agent'),
	([1, 2], [[2]], 'Overall', 'y_values must have self.episodes / self.plot_interval many items')
]


@pytest.mark.parametrize('x_values, y_values, plot_type, expected_message', incorrect_create_line_plot_testcases)
def test_incorrect_create_line_plot(x_values, y_values, plot_type, expected_message):
	monitor.configurator.setup_monitoring(episodes=4, plot_interval=2)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.evaluator._create_line_plot(x_values, y_values, 'test_plot', plot_type)
	assert expected_message in str(assertion_message.value)


def test_incorrect_create_line_plot_runtime_errors():
	monitor.configurator.setup_monitoring(episodes=4, plot_interval=2)
	with pytest.raises(RuntimeError) as assertion_message:
		monitor.evaluator._create_line_plot([1, 2], [[1, 3]], 'test_plot', 'Unknown_metric_type')
	assert 'this metric_type is unknown: Unknown_metric_type' in str(assertion_message.value)
