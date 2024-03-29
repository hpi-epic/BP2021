import os
from unittest.mock import patch

import numpy as np
import pytest
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.monitoring.agent_monitoring.am_monitoring as monitoring
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import FixedPriceCEAgent, RuleBasedCEAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

monitor = monitoring.Monitor()

config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceMonopoly)


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(
		separate_markets=False,
		episodes=50,
		plot_interval=10,
		marketplace=circular_market.CircularEconomyMonopoly,
		agents=[(QLearningAgent, [os.path.join(PathManager.data_path, 'CircularEconomyMonopoly_QLearningAgent.dat')])],
		config_market=config_market)


evaluate_session_testcases = [
	([(RuleBasedCEAgent, [])], circular_market.CircularEconomyMonopoly,
		{'profits/all': [[5, 10, 0, -5]], 'a/reward': [10, 15, 25, -50]}),
	([(RuleBasedCEAgent, []), (FixedPriceCEAgent, [])], circular_market.CircularEconomyDuopoly,
		{'profits/all': [[5, 10, 0, -5], [5, -10, 60, 5]], 'a/reward': [[10, 15, 25, -50], [10, 15, 25, -50]]})
]


@pytest.mark.parametrize('agents, marketplace, rewards', evaluate_session_testcases)
def test_evaluate_session(agents, marketplace, rewards):
	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.clf'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.xlabel'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.title'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.legend'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.pause'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.draw'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.savefig'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		monitor.configurator.setup_monitoring(
			episodes=4,
			plot_interval=1,
			separate_markets=False,
			agents=agents,
			marketplace=marketplace,
			config_market=config_market)
		monitor.evaluator.evaluate_session(rewards)


# all arrays in rewards must be of the same size
def test_rewards_array_size():
	# Numpy doesn't like nested arrays of different sizes, need to specify dtype=object
	rewards_wrong = np.array([[1, 2], [1, 2, 3]], dtype=object)

	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt'):
		with pytest.raises(AssertionError) as assertion_message:
			monitor.evaluator.create_histogram(rewards_wrong, True)
		assert 'all rewards-arrays must be of the same size' in str(assertion_message.value)


create_histogram_statistics_plots_testcases = [
	([(RuleBasedCEAgent, [])], [[100, 0]], 5, [(1.0, 0.0, 0.0, 1.0)], (0.0, 1000.0)),
	([(RuleBasedCEAgent, []), (RuleBasedCEAgent, [])],
		[[100, 0], [10, 5]], 5, [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0)], (0.0, 1000.0)),
	([(RuleBasedCEAgent, []), (RuleBasedCEAgent, []), (RuleBasedCEAgent, []), (RuleBasedCEAgent, [])],
		[[100, 0], [10, 5], [100, 10000], [10, 1000]],
		50,
		[(1.0, 0.0, 0.0, 1.0), (0.5234360234360234, 1.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0), (0.4296860234360234, 0.0, 1.0, 1.0)],
		(0.0, 10000.0))
]


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', create_histogram_statistics_plots_testcases)
def test_create_histogram(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.configurator.setup_monitoring(separate_markets=True, agents=agents, config_market=config_market)
	name_list = [agent.name for agent in monitor.configurator.agents]
	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.clf'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.xlabel'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.title'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.hist') as hist_mock, \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.legend') as legend_mock, \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.pause'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.savefig') as save_mock, \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.evaluator.create_histogram(rewards, True)
		hist_mock.assert_called_once_with(rewards, bins=plot_bins, color=agent_color, rwidth=0.9, range=lower_upper_range, edgecolor='black')
		legend_mock.assert_called_once_with(name_list)
		save_mock.assert_called_once_with(fname=os.path.join(monitor.configurator.folder_path, 'default_histogram.svg'), transparent=True)


def test_create_histogram_without_saving_to_directory():
	monitor.configurator.setup_monitoring(separate_markets=False, agents=[(RuleBasedCEAgent, [])], config_market=config_market)
	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.clf'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.xlabel'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.title'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.hist'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.legend'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.pause'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.draw'), \
		patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt.savefig') as save_mock, \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.evaluator.create_histogram([[100, 0]], False)
		save_mock.assert_not_called()


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', create_histogram_statistics_plots_testcases)
def test_create_statistics_plots(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.configurator.setup_monitoring(
		separate_markets=True,
		agents=agents,
		episodes=len(rewards[0]),
		plot_interval=1,
		config_market=config_market)
	with patch('recommerce.monitoring.agent_monitoring.am_evaluation.plt'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'), \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.evaluator._create_statistics_plots(rewards)


incorrect_create_line_plot_testcases = [
	([1, 2, 3], [[2], [1]], 'Overall', 'x_values must have 2 items, had 3'),
	([1, 2], [[2, 1], [1, 2]], 'Overall', 'y_values must have one entry per agent'),
	([1, 2], [[2]], 'Overall', 'Each value in y_values must have 2 items, was [[2]]')
]


@pytest.mark.parametrize('x_values, y_values, plot_type, expected_message', incorrect_create_line_plot_testcases)
def test_incorrect_create_line_plot(x_values, y_values, plot_type, expected_message):
	monitor.configurator.setup_monitoring(episodes=4, plot_interval=2, config_market=config_market)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.evaluator._create_line_plot(x_values, y_values, 'test_plot', plot_type)
	assert expected_message in str(assertion_message.value)


def test_incorrect_create_line_plot_runtime_errors():
	monitor.configurator.setup_monitoring(episodes=4, plot_interval=2, config_market=config_market)
	with pytest.raises(RuntimeError) as assertion_message:
		monitor.evaluator._create_line_plot([1, 2], [[1, 3]], 'test_plot', 'Unknown_metric_type')
	assert 'this metric_type is unknown: Unknown_metric_type' in str(assertion_message.value)
