import os
import re
import shutil

import numpy as np
import pytest

import agents.vendors as vendors
import market.sim_market as sim_market
import monitoring.agent_monitoring as am
import tests.utils_tests as ut_t
from monitoring.agent_monitoring import Monitor

monitor = Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = Monitor()
	monitor.setup_monitoring(enable_live_draw=False, subfolder_name='test_plots_' + function.__name__)


def teardown_module(module):
	print('***TEARDOWN***')
	for f in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring/'):
		if re.match('test_plots_*', f):
			shutil.rmtree(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring/' + f)


def test_init_default_values():
	test_monitor = am.Monitor()
	assert test_monitor.enable_live_draw is True
	assert 500 == test_monitor.episodes
	assert 50 == test_monitor.plot_interval
	assert isinstance(test_monitor.marketplace, sim_market.CircularEconomyMonopolyScenario)
	assert isinstance(test_monitor.agents[0], vendors.QLearningCEAgent)
	assert 1 == len(test_monitor.agents)
	assert ['#0000ff'] == test_monitor.agent_colors
	assert test_monitor.subfolder_name.startswith('plots_')
	assert os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + test_monitor.subfolder_name) == os.path.normcase(test_monitor.folder_path)


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
	assert 'subfoldername' == monitor.subfolder_name
	assert 2 == len(monitor.agent_colors)


def test_incorrect_setup_monitoring():
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(enable_live_draw=1)
	assert 'enable_live_draw must be a Boolean' in str(assertion_message.value)

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(episodes='Hello World')
	assert 'episodes must be of type int' in str(assertion_message.value)

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(plot_interval='1')
	assert 'plot_interval must be of type int' in str(assertion_message.value)

	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(marketplace=vendors.RuleBasedCEAgent)
	assert 'the marketplace must be a subclass of SimMarket' in str(assertion_message.value)
	with pytest.raises(TypeError):
		monitor.setup_monitoring(marketplace=sim_market.ClassicScenario())

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


def test_mismatched_scenarios():
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(marketplace=sim_market.ClassicScenario, agents=[(vendors.RuleBasedCEAgent, [])])
	assert 'the agent and marketplace must be of the same economy type' in str(assertion_message.value)


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


def test_get_modelfile_path():
	with pytest.raises(AssertionError) as assertion_message:
		monitor.get_modelfile_path('wrong_extension.png')
	assert 'the modelfile must be a .dat file' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.get_modelfile_path('non_existing_modelfile.dat')
	assert 'the specified modelfile does not exist' in str(assertion_message.value)


# Test once for a Linear, Circular and RebuyPrice Economy
def test_get_action_space():
	monitor.setup_monitoring(agents=[(vendors.QLearningLEAgent, [])], marketplace=sim_market.ClassicScenario)
	monitor.setup_monitoring(agents=[(vendors.QLearningCEAgent, [])], marketplace=sim_market.CircularEconomyMonopolyScenario)
	monitor.setup_monitoring(agents=[(vendors.QLearningCERebuyAgent, [])], marketplace=sim_market.CircularEconomyRebuyPriceMonopolyScenario)


def test_setting_market_not_agents():
	monitor.setup_monitoring(marketplace=sim_market.CircularEconomyMonopolyScenario)


def test_setup_with_invalid_agents():
	with pytest.raises(AssertionError):
		monitor.setup_monitoring(agents=[vendors.FixedPriceLEAgent, vendors.FixedPriceCERebuyAgent])


def test_setup_with_valid_agents():
	monitor.setup_monitoring(agents=[(vendors.FixedPriceCERebuyAgent, []), (vendors.FixedPriceCEAgent, [])])


def test_metrics_average():
	assert 6 == monitor.metrics_average(ut_t.create_mock_rewards(12))


def test_metrics_median():
	assert 6 == monitor.metrics_median(ut_t.create_mock_rewards(12))


def test_metrics_maximum():
	assert 11 == monitor.metrics_maximum(ut_t.create_mock_rewards(12))


def test_metrics_minimum():
	assert 1 == monitor.metrics_minimum(ut_t.create_mock_rewards(12))


def test_round_up():
	assert monitor.round_up(999, -3) == 1000


def test_round_down():
	assert monitor.round_down(999, -3) == 0


# all arrays in rewards must be of the same size
def test_rewards_array_size():
	# Numpy doesn't like nested arrays of different sizes, need to specify dtype=object
	rewards_wrong = np.array([[1, 2], [1, 2, 3]], dtype=object)
	with pytest.raises(Exception):
		monitor.create_histogram(rewards_wrong)


# def test_get_episode_reward():
# 	json = ut_t.create_mock_json_sim_market(episode_size='2')
# 	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
# 		ut_t.check_mock_file_sim_market(mock_file, json)
# 		reload(ut)
# 		all_steps_reward = [[1, 2, 3, 4], [4, 5, 6, 7], [1, 3, 4, 5]]
# 		assert [[3, 7], [9, 13], [4, 9]] == monitor.get_episode_rewards(all_steps_reward)
	# reload(ut)


agent_rewards_histogram = [
	([(vendors.RuleBasedCEAgent, [])], [[100, 0]]),
	([(vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, [])], [[100, 0], [10, 5]]),
	([(vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, []), (vendors.RuleBasedCEAgent, [])],
		[[100, 0], [10, 5], [100, 10000], [10, 1000]])
]


@pytest.mark.parametrize('agents, rewards', agent_rewards_histogram)
def test_create_histogram(agents, rewards):
	monitor.setup_monitoring(agents=agents)
	monitor.create_histogram(rewards)


@pytest.mark.parametrize('agents, rewards', agent_rewards_histogram)
def test_create_statistics_plots(agents, rewards):
	monitor.setup_monitoring(agents=agents, episodes=len(rewards[0]), plot_interval=1)
	monitor.create_statistics_plots(rewards)


def test_create_line_plot():
	monitor.setup_monitoring(episodes=4, plot_interval=2)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.create_line_plot([1, 2, 3], [[2], [1]])
	assert 'x_values must have self.episodes / self.plot_interval many items' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.create_line_plot([1, 2], [[2], [1]])
	assert 'y_values must have one entry per agent' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.create_line_plot([1, 2], [[2]])
	assert 'y_values must have self.episodes / self.plot_interval many items' in str(assertion_message.value)


def test_run_marketplace():
	monitor.setup_monitoring(episodes=100, plot_interval=100, agents=[(vendors.FixedPriceCEAgent, [(5, 2)])])
	agent_rewards = monitor.run_marketplace()
	print(agent_rewards)
	assert 1 == len(monitor.agents)
	assert monitor.episodes == len(agent_rewards[0])


def test_main():
	monitor.setup_monitoring(enable_live_draw=False, episodes=10, plot_interval=10, subfolder_name='test_plots_')
	current_configuration = monitor.get_configuration()
	am.main(monitor)
	assert current_configuration == monitor.get_configuration(), 'the monitor configuration should not be changed within main'
	assert os.path.exists(monitor.folder_path)
