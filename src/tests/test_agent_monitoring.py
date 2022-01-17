import os
from unittest.mock import patch

import numpy as np
import pytest

import agents.vendors as vendors
import market.sim_market as sim_market
import monitoring.agent_monitoring as am
from monitoring.agent_monitoring import Monitor

monitor = Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = Monitor()
	monitor.setup_monitoring(enable_live_draw=False, subfolder_name='test_plots_' + function.__name__)


def test_init_default_values():
	test_monitor = am.Monitor()
	assert test_monitor.enable_live_draw is True
	assert 500 == test_monitor.episodes
	assert 50 == test_monitor.plot_interval
	assert isinstance(test_monitor.marketplace, sim_market.CircularEconomyMonopolyScenario)
	assert isinstance(test_monitor.agents[0], vendors.QLearningCEAgent)
	assert 1 == len(test_monitor.agents)
	assert [(0.0, 0.0, 1.0, 1.0)] == test_monitor.agent_colors
	# folder_path can hardly be tested due to the default involving the current DateTime


def test_get_folder():
	# if you change the name of this function, change it in the assert as well!
	monitor._get_folder()
	assert os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'monitoring', 'test_plots_test_get_folder')))


def test_get_modelfile_path():
	with patch('monitoring.agent_monitoring.os.path.exists') as mock_exists:
		mock_exists.return_value = False
		with pytest.raises(AssertionError) as assertion_message:
			monitor._get_modelfile_path('non_existing_modelfile')
		assert 'the specified modelfile does not exist' in str(assertion_message.value)


# Test once for a Linear, Circular and RebuyPrice Economy
vendor_market = [
	((vendors.QLearningLEAgent, ['ClassicScenario_QLearningLEAgent']), sim_market.ClassicScenario),
	((vendors.QLearningCEAgent, ['CircularEconomy_QLearningCEAgent']), sim_market.CircularEconomyMonopolyScenario),
	((vendors.QLearningCERebuyAgent, ['CircularEconomyRebuyPrice_QLearningCERebuyAgent']), sim_market.CircularEconomyRebuyPriceMonopolyScenario)
]


@pytest.mark.parametrize('agent, market', vendor_market)
def test_get_action_space(agent, market):
	monitor.setup_monitoring(agents=[agent], marketplace=market)


def test_update_agents():
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, ['modelfile.dat', 'arg', 'too_much'])])
	assert 'the argument list for a RL-agent must have length between 0 and 2' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, [1, 2, 3, 4])])
	assert 'the argument list for a RL-agent must have length between 0 and 2' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, ['modelfile.dat', 35])])
	assert 'the arguments for a RL-agent must be of type str' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, [25])])
	assert 'the arguments for a RL-agent must be of type str' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, ['agent_name', 'modelfile.dat'])])
	assert 'if two arguments are provided, the first one must be the modelfile.' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, ['mymodel.dat'])])
	assert 'the specified modelfile does not exist' in str(assertion_message.value)
	# some valid options that should pass
	monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, [])])
	monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, ['new_name'])])
	monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat'])])
	monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'new_name'])])
	monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, [f'{type(monitor.marketplace).__name__}_{vendors.QLearningCEAgent.__name__}.dat'])])


def test_correct_setup_monitoring():
	monitor.setup_monitoring(enable_live_draw=False, episodes=10, plot_interval=2, marketplace=sim_market.CircularEconomyMonopolyScenario, agents=[(vendors.HumanPlayerCERebuy, ['reptiloid']), (vendors.QLearningCERebuyAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'q_learner'])], subfolder_name='subfoldername')
	assert monitor.enable_live_draw is False
	assert 10 == monitor.episodes
	assert 2 == monitor.plot_interval
	assert isinstance(monitor.marketplace, sim_market.CircularEconomyMonopolyScenario)
	assert 2 == len(monitor.agents)
	assert isinstance(monitor.agents[0], vendors.HumanPlayerCERebuy)
	assert isinstance(monitor.agents[1], vendors.QLearningCERebuyAgent)
	assert 'reptiloid' == monitor.agents[0].name
	assert 'q_learner' == monitor.agents[1].name
	assert os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'monitoring', 'subfoldername'))) == os.path.normcase(os.path.abspath(monitor.folder_path))
	assert 2 == len(monitor.agent_colors)


def test_setting_multiple_agents():
	monitor.setup_monitoring(agents=[(vendors.FixedPriceCERebuyAgent, []), (vendors.FixedPriceCEAgent, [])])
	monitor.setup_monitoring(agents=[(vendors.FixedPriceCERebuyAgent, []), (vendors.FixedPriceCEAgent, []), (vendors.FixedPriceCERebuyAgent, [])])


def test_setting_market_not_agents():
	monitor.setup_monitoring(marketplace=sim_market.CircularEconomyMonopolyScenario)


def test_incorrect_setup_monitoring():
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(enable_live_draw=1)
	assert 'enable_live_draw must be a Boolean' in str(assertion_message.value)

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(episodes='Hello World')
	assert 'episodes must be of type int' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(episodes=0)
	assert 'episodes must not be 0' in str(assertion_message.value)

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(plot_interval='1')
	assert 'plot_interval must be of type int' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(plot_interval=0)
	assert 'plot_interval must not be 0' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(episodes=4, plot_interval=5)
	assert 'plot_interval must be <= episodes, or no plots can be generated.' in str(assertion_message.value)

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(marketplace=vendors.RuleBasedCEAgent)
	assert 'the marketplace must be a subclass of SimMarket' in str(assertion_message.value)
	with pytest.raises(TypeError):
		monitor.setup_monitoring(marketplace=sim_market.ClassicScenario())
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(marketplace=sim_market.ClassicScenario, agents=[(vendors.RuleBasedCEAgent, [])])
	assert 'the agent and marketplace must be of the same economy type' in str(assertion_message.value)

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[vendors.RuleBasedCEAgent])
	assert 'agents must be a list of tuples' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[[vendors.RuleBasedCEAgent, 1, '2']])
	assert 'agents must be a list of tuples' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.RuleBasedCEAgent)])
	assert 'agents must be a list of tuples' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.RuleBasedCEAgent, ['arg'], 'too_much')])
	assert 'the list entries in agents must have size 2 ([agent_class, arguments])' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(sim_market.ClassicScenario, [])])
	assert 'the first entry in each agent-tuple must be an agent class in `vendors.py`' in str(assertion_message.value)
	with pytest.raises(TypeError) as assertion_message:
		monitor.setup_monitoring(agents=[(sim_market.ClassicScenario(), [])])
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.RuleBasedCEAgent, sim_market.ClassicScenario)])
	assert 'the second entry in each agent-tuple must be a list' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.RuleBasedCEAgent, 'new_name')])
	assert 'the second entry in each agent-tuple must be a list' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.RuleBasedCEAgent, []), (vendors.FixedPriceLEAgent, [])])
	assert 'the agents must all be of the same type (Linear/Circular)' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.RuleBasedCEAgent, []), (vendors.FixedPriceLEAgent, []), (vendors.FixedPriceCEAgent, [])])
	assert 'the agents must all be of the same type (Linear/Circular)' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.FixedPriceLEAgent, [])], marketplace=sim_market.CircularEconomyRebuyPriceMonopolyScenario)
	assert 'the agent and marketplace must be of the same economy type (Linear/Circular)' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[(vendors.FixedPriceCEAgent, [])], marketplace=sim_market.ClassicScenario)
	assert 'the agent and marketplace must be of the same economy type (Linear/Circular)' in str(assertion_message.value)
	with pytest.raises(TypeError):
		monitor.setup_monitoring(agents=[(vendors.RuleBasedCEAgent(), [])])

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(subfolder_name=1)
	assert 'subfolder_name must be of type str' in str(assertion_message.value)


def test_get_configuration():
	monitor.setup_monitoring(enable_live_draw=False, episodes=10, plot_interval=2, marketplace=sim_market.CircularEconomyMonopolyScenario, agents=[(vendors.HumanPlayerCERebuy, ['reptiloid']), (vendors.QLearningCERebuyAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'q_learner'])], subfolder_name='subfoldername')
	current_configuration = monitor.get_configuration()
	assert len(current_configuration) == 7, 'parameters were updated in agent_monitoring.py, but not updated in the tests!'
	assert 'enable_live_draw' in current_configuration
	assert 'episodes' in current_configuration
	assert 'plot_interval' in current_configuration
	assert 'marketplace' in current_configuration
	assert 'agents' in current_configuration
	assert 'agent_colors' in current_configuration
	assert 'folder_path' in current_configuration


def test_setup_with_invalid_agents():
	with pytest.raises(AssertionError):
		monitor.setup_monitoring(agents=[vendors.FixedPriceLEAgent, vendors.FixedPriceCERebuyAgent])


# all arrays in rewards must be of the same size
def test_rewards_array_size():
	# Numpy doesn't like nested arrays of different sizes, need to specify dtype=object
	rewards_wrong = np.array([[1, 2], [1, 2, 3]], dtype=object)

	with patch('monitoring.agent_monitoring.plt'):
		with pytest.raises(AssertionError) as assertion_message:
			monitor.create_histogram(rewards_wrong)
		assert 'all rewards-arrays must be of the same size' in str(assertion_message.value)


agent_rewards_histogram = [
	([(vendors.RuleBasedCEAgent, [])], [[100, 0]], 1, [(1.0, 0.0, 0.0, 1.0)], (0.0, 1000.0)),
	([(vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, [])], [[100, 0], [10, 5]], 1, [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0)], (0.0, 1000.0)),
	([(vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, [])],
		[[100, 0], [10, 5], [100, 10000], [10, 1000]],
		10, [(1.0, 0.0, 0.0, 1.0), (0.5234360234360234, 1.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0), (0.4296860234360234, 0.0, 1.0, 1.0)], (0.0, 10000.0))
]


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', agent_rewards_histogram)
def test_create_histogram(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.setup_monitoring(agents=agents, enable_live_draw=True)
	name_list = [a.name for a in monitor.agents]
	with patch('monitoring.agent_monitoring.plt.clf'), \
		patch('monitoring.agent_monitoring.plt.xlabel'), \
		patch('monitoring.agent_monitoring.plt.title'), \
		patch('monitoring.agent_monitoring.plt.hist') as hist_mock, \
		patch('monitoring.agent_monitoring.plt.legend') as legend_mock, \
		patch('monitoring.agent_monitoring.plt.pause'), \
		patch('monitoring.agent_monitoring.plt.draw') as draw_mock, \
		patch('monitoring.agent_monitoring.plt.savefig') as save_mock, \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.create_histogram(rewards)
		hist_mock.assert_called_once_with(rewards, bins=plot_bins, color=agent_color, rwidth=0.9, range=lower_upper_range, edgecolor='black')
		legend_mock.assert_called_once_with(name_list)
		draw_mock.assert_called_once()
		save_mock.assert_called_once_with(fname=os.path.join(monitor.folder_path, 'histograms', 'default.svg'))


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', agent_rewards_histogram)
def test_create_statistics_plots(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.setup_monitoring(agents=agents, episodes=len(rewards[0]), plot_interval=1)
	with patch('monitoring.agent_monitoring.plt'), \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.create_statistics_plots(rewards)


def test_incorrect_create_line_plot():
	monitor.setup_monitoring(episodes=4, plot_interval=2)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.create_line_plot([1, 2, 3], [[2], [1]], 'test_plot', 'Overall')
	assert 'x_values must have self.episodes / self.plot_interval many items' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.create_line_plot([1, 2], [[2], [1]], 'test_plot', 'Overall')
	assert 'y_values must have one entry per agent' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.create_line_plot([1, 2], [[2]], 'test_plot', 'Overall')
	assert 'y_values must have self.episodes / self.plot_interval many items' in str(assertion_message.value)
	with pytest.raises(RuntimeError) as assertion_message:
		monitor.create_line_plot([1, 2], [[1, 3]], 'test_plot', 'Unknown_metric_type')
	assert 'this metric_type is unknown: Unknown_metric_type' in str(assertion_message.value)


def test_run_marketplace():
	monitor.setup_monitoring(episodes=100, plot_interval=100, agents=[(vendors.FixedPriceCEAgent, [(5, 2)])])
	with patch('monitoring.agent_monitoring.plt'), \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		agent_rewards = monitor.run_marketplace()
		assert 1 == len(monitor.agents)
		assert monitor.episodes == len(agent_rewards[0])


def test_run_monitoring_session():
	monitor.setup_monitoring(episodes=10, plot_interval=10)
	current_configuration = monitor.get_configuration()
	with patch('monitoring.agent_monitoring.plt'), \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		am.run_monitoring_session(monitor)
		assert current_configuration == monitor.get_configuration(), 'the monitor configuration should not be changed within run_monitoring()'
		assert os.path.exists(monitor.folder_path)


@patch('monitoring.agent_monitoring.input', create=True)
def test_run_monitoring_ratio(mocked_input):
	# ratio is over 50, program should ask if we want to continue. We answer 'no'
	with patch('monitoring.agent_monitoring.plt'), \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		mocked_input.side_effect = ['n']
		monitor.setup_monitoring(episodes=51, plot_interval=1)
		am.run_monitoring_session(monitor)
