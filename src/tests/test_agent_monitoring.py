import os
import shutil
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
	monitor.setup_monitoring(enable_live_draw=False, subfolder_name=f'test_plots_{function.__name__}')


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
	# if you change the name of this function, change it here as well!
	foldername = 'test_plots_test_get_folder'
	monitor._get_folder()
	assert os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'monitoring', foldername)))
	shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'monitoring', foldername))


def test_get_modelfile_path():
	with patch('monitoring.agent_monitoring.os.path.exists') as mock_exists:
		mock_exists.return_value = False
		with pytest.raises(AssertionError) as assertion_message:
			monitor._get_modelfile_path('non_existing_modelfile')
		assert 'the specified modelfile does not exist' in str(assertion_message.value)


incorrect_update_agents_testcases = [
	([(vendors.QLearningCEAgent, ['modelfile.dat', 'arg', 'too_much'])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(vendors.QLearningCEAgent, [1, 2, 3, 4])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(vendors.QLearningCEAgent, ['modelfile.dat', 35])], 'the arguments for a RL-agent must be of type str'),
	([(vendors.QLearningCEAgent, [25])], 'the arguments for a RL-agent must be of type str'),
	([(vendors.QLearningCEAgent, ['agent_name', 'modelfile.dat'])], 'if two arguments are provided, the first one must be the modelfile.'),
	([(vendors.QLearningCEAgent, ['mymodel.dat'])], 'the specified modelfile does not exist')
]


@pytest.mark.parametrize('agents, expected_message', incorrect_update_agents_testcases)
def test_incorrect_update_agents(agents, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=agents)
	assert expected_message in str(assertion_message.value)


correct_update_agents_testcases = [
	[(vendors.QLearningCEAgent, [])],
	[(vendors.QLearningCEAgent, ['new_name'])],
	[(vendors.QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat'])],
	[(vendors.QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'new_name'])],
	[(vendors.QLearningCEAgent, [f'{type(monitor.marketplace).__name__}_{vendors.QLearningCEAgent.__name__}.dat'])]
]


@pytest.mark.parametrize('agents', correct_update_agents_testcases)
def test_correct_update_agents(agents):
	monitor.setup_monitoring(agents=agents)


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


setting_multiple_agents_testcases = [
	[(vendors.FixedPriceCERebuyAgent, []), (vendors.FixedPriceCEAgent, [])],
	[(vendors.FixedPriceCERebuyAgent, []), (vendors.FixedPriceCEAgent, []), (vendors.FixedPriceCERebuyAgent, [])]
]


@pytest.mark.parametrize('agents', setting_multiple_agents_testcases)
def test_setting_multiple_agents(agents):
	monitor.setup_monitoring(agents=agents)


def test_setting_market_not_agents():
	monitor.setup_monitoring(marketplace=sim_market.CircularEconomyMonopolyScenario)


incorrect_setup_monitoring_testcases = [
	({'enable_live_draw': 1}, 'enable_live_draw must be a Boolean'),
	({'episodes': 'Hello World'}, 'episodes must be of type int'),
	({'episodes': 0}, 'episodes must not be 0'),
	({'plot_interval': '1'}, 'plot_interval must be of type int'),
	({'plot_interval': 0}, 'plot_interval must not be 0'),
	({'episodes': 4, 'plot_interval': 5}, 'plot_interval must be <= episodes, or no plots can be generated.'),
	({'marketplace': vendors.RuleBasedCEAgent}, 'the marketplace must be a subclass of SimMarket'),
	({'marketplace': sim_market.ClassicScenario, 'agents': [(vendors.RuleBasedCEAgent, [])]}, 'the agent and marketplace must be of the same economy type'),
	({'agents': [vendors.RuleBasedCEAgent]}, 'agents must be a list of tuples'),
	({'agents': [[vendors.RuleBasedCEAgent, 1, '2']]}, 'agents must be a list of tuples'),
	({'agents': [(vendors.RuleBasedCEAgent)]}, 'agents must be a list of tuples'),
	({'agents': [vendors.FixedPriceLEAgent, vendors.FixedPriceCERebuyAgent]}, 'agents must be a list of tuples'),
	({'agents': [(vendors.RuleBasedCEAgent, ['arg'], 'too_much')]}, 'the list entries in agents must have size 2 ([agent_class, arguments])'),
	({'agents': [(sim_market.ClassicScenario, [])]}, 'the first entry in each agent-tuple must be an agent class in `vendors.py`'),
	({'agents': [(vendors.RuleBasedCEAgent, sim_market.ClassicScenario)]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(vendors.RuleBasedCEAgent, 'new_name')]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(vendors.RuleBasedCEAgent, []), (vendors.FixedPriceLEAgent, [])]}, 'the agents must all be of the same type (Linear/Circular)'),
	({'agents': [(vendors.RuleBasedCEAgent, []), (vendors.FixedPriceLEAgent, []), (vendors.FixedPriceCEAgent, [])]}, 'the agents must all be of the same type (Linear/Circular)'),
	({'marketplace': sim_market.CircularEconomyRebuyPriceMonopolyScenario, 'agents': [(vendors.FixedPriceLEAgent, [])]}, 'the agent and marketplace must be of the same economy type (Linear/Circular)'),
	({'marketplace': sim_market.ClassicScenario, 'agents': [(vendors.FixedPriceCEAgent, [])]}, 'the agent and marketplace must be of the same economy type (Linear/Circular)'),
	({'subfolder_name': 1}, 'subfolder_name must be of type str')
]


@pytest.mark.parametrize('parameters, expected_message', incorrect_setup_monitoring_testcases)
def test_incorrect_setup_monitoring(parameters, expected_message):
	dict = {
		'enable_live_draw': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None,
		'subfolder_name': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(
			enable_live_draw=dict['enable_live_draw'],
			episodes=dict['episodes'],
			plot_interval=dict['plot_interval'],
			marketplace=dict['marketplace'],
			agents=dict['agents'],
			subfolder_name=dict['subfolder_name']
		)
	assert expected_message in str(assertion_message.value)


incorrect_setup_monitoring_type_errors_testcases = [
	{'marketplace': sim_market.ClassicScenario()},
	{'agents': [(sim_market.ClassicScenario(), [])]},
	{'agents': [(vendors.RuleBasedCEAgent(), [])]}
]


@pytest.mark.parametrize('parameters', incorrect_setup_monitoring_type_errors_testcases)
def test_incorrect_setup_monitoring_type_errors(parameters):
	dict = {
		'enable_live_draw': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None,
		'subfolder_name': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	with pytest.raises(TypeError):
		monitor.setup_monitoring(
			enable_live_draw=dict['enable_live_draw'],
			episodes=dict['episodes'],
			plot_interval=dict['plot_interval'],
			marketplace=dict['marketplace'],
			agents=dict['agents'],
			subfolder_name=dict['subfolder_name']
		)


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


# all arrays in rewards must be of the same size
def test_rewards_array_size():
	# Numpy doesn't like nested arrays of different sizes, need to specify dtype=object
	rewards_wrong = np.array([[1, 2], [1, 2, 3]], dtype=object)

	with patch('monitoring.agent_monitoring.plt'):
		with pytest.raises(AssertionError) as assertion_message:
			monitor.create_histogram(rewards_wrong)
		assert 'all rewards-arrays must be of the same size' in str(assertion_message.value)


create_histogram_statistics_plots_testcases = [
	([(vendors.RuleBasedCEAgent, [])], [[100, 0]], 1, [(1.0, 0.0, 0.0, 1.0)], (0.0, 1000.0)),
	([(vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, [])], [[100, 0], [10, 5]], 1, [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0)], (0.0, 1000.0)),
	([(vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, [])],
		[[100, 0], [10, 5], [100, 10000], [10, 1000]],
		10, [(1.0, 0.0, 0.0, 1.0), (0.5234360234360234, 1.0, 0.0, 1.0), (0.0, 1.0, 0.9531223422015865, 1.0), (0.4296860234360234, 0.0, 1.0, 1.0)], (0.0, 10000.0))
]


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', create_histogram_statistics_plots_testcases)
def test_create_histogram(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.setup_monitoring(enable_live_draw=True, agents=agents)
	name_list = [agent.name for agent in monitor.agents]
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


@pytest.mark.parametrize('agents, rewards, plot_bins, agent_color, lower_upper_range', create_histogram_statistics_plots_testcases)
def test_create_statistics_plots(agents, rewards, plot_bins, agent_color, lower_upper_range):
	monitor.setup_monitoring(agents=agents, episodes=len(rewards[0]), plot_interval=1)
	with patch('monitoring.agent_monitoring.plt'), \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		exists_mock.return_value = True

		monitor.create_statistics_plots(rewards)


incorrect_create_line_plot_testcases = [
	([1, 2, 3], [[2], [1]], 'Overall', 'x_values must have self.episodes / self.plot_interval many items'),
	([1, 2], [[2], [1]], 'Overall', 'y_values must have one entry per agent'),
	([1, 2], [[2]], 'Overall', 'y_values must have self.episodes / self.plot_interval many items')
]


@pytest.mark.parametrize('x_values, y_values, plot_type, expected_message', incorrect_create_line_plot_testcases)
def test_incorrect_create_line_plot(x_values, y_values, plot_type, expected_message):
	monitor.setup_monitoring(episodes=4, plot_interval=2)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.create_line_plot(x_values, y_values, 'test_plot', plot_type)
	assert expected_message in str(assertion_message.value)


def test_incorrect_create_line_plot_runtime_errors():
	monitor.setup_monitoring(episodes=4, plot_interval=2)
	with pytest.raises(RuntimeError) as assertion_message:
		monitor.create_line_plot([1, 2], [[1, 3]], 'test_plot', 'Unknown_metric_type')
	assert 'this metric_type is unknown: Unknown_metric_type' in str(assertion_message.value)


def test_run_marketplace():
	monitor.setup_monitoring(episodes=100, plot_interval=100, agents=[(vendors.FixedPriceCEAgent, [(5, 2)])])
	with patch('monitoring.agent_monitoring.plt'), \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		exists_mock.return_value = True
		agent_rewards = monitor.run_marketplace()
		assert 1 == len(agent_rewards)
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


def test_run_monitoring_ratio():
	# ratio is over 50, program should ask if we want to continue. We answer 'no'
	with patch('monitoring.agent_monitoring.plt'), \
		patch('monitoring.agent_monitoring.input', create=True) as mocked_input, \
		patch('monitoring.agent_monitoring.os.path.exists') as exists_mock:
		mocked_input.side_effect = ['n']
		exists_mock.return_value = True
		monitor.setup_monitoring(episodes=51, plot_interval=1)
		am.run_monitoring_session(monitor)
