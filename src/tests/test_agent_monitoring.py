import os
import re
import shutil

import numpy as np
import pytest

from .context import Monitor, agent
from .context import agent_monitoring as am
from .context import sim_market
from .context import utils as ut

monitor = Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = Monitor()
	monitor.setup_monitoring(draw_enabled=False, subfolder_name='test_plots_')


# teardown after each test
def teardown_function(function):
	print('***TEARDOWN***')
	for f in os.listdir('./monitoring'):
		if re.match('test_plots_*', f):
			shutil.rmtree('./monitoring/' + f)


# create mock rewards list
def create_mock_rewards() -> list:
	mock_rewards = []
	for number in range(1, 12):
		mock_rewards.append(number)
	return mock_rewards


def test_correct_setup_monitoring():
	monitor.setup_monitoring(draw_enabled=False, episodes=10, plot_interval=2, modelfile='modelfile.dat', marketplace=sim_market.CircularEconomy, agents=[agent.HumanPlayerCERebuy], subfolder_name='subfoldername')
	assert monitor.enable_live_draws is False
	assert 10 == monitor.episodes
	assert 2 == monitor.plot_interval
	assert os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + 'modelfile.dat' == monitor.path_to_modelfile
	assert isinstance(monitor.marketplace, sim_market.CircularEconomy)
	assert all(isinstance(test_agent, agent.HumanPlayerCERebuy) for test_agent in monitor.agents)
	assert 'subfoldername' == monitor.subfolder_name
	assert 1 == len(monitor.agent_colors)


def test_incorrect_setup_monitoring():
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(draw_enabled=1)
	assert 'draw_enabled must be True or False' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(episodes='Hello World')
	assert 'episodes must be of type int' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(plot_interval='1')
	assert 'plot_interval must be of type int' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(modelfile=1)
	assert 'modelfile must be of type string' in str(assertion_message.value)
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(marketplace=agent.RuleBasedCEAgent)
	assert 'the marketplace must be a subclass of sim_market' in str(assertion_message.value)
	with pytest.raises(TypeError):
		monitor.setup_monitoring(marketplace=sim_market.ClassicScenario())
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(agents=[sim_market.ClassicScenario])
	assert 'the agents must be agent classes in agent.py' in str(assertion_message.value)
	with pytest.raises(TypeError):
		monitor.setup_monitoring(agents=[agent.RuleBasedCEAgent()])
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(subfolder_name=1)
	assert 'subfolder_name must be of type string' in str(assertion_message.value)


def test_mismatched_scenarios():
	with pytest.raises(AssertionError) as assertion_message:
		monitor.setup_monitoring(marketplace=sim_market.ClassicScenario, agents=[agent.RuleBasedCEAgent])
	assert 'the agent and marketplace must be of the same economy type' in str(assertion_message.value)


def test_mismatched_modelfile():
	with pytest.raises(RuntimeError) as assertion_message:
		monitor.setup_monitoring(modelfile='QLearningAgent_ClassicScenario.dat', agents=[agent.QLearningCEAgent], marketplace=sim_market.CircularEconomyRebuyPrice)
	assert 'the modelfile is not compatible with the agent you tried to instantiate' in str(assertion_message.value)


def test_init_default_values():
	test_monitor = am.Monitor()
	assert test_monitor.enable_live_draws is True
	assert 500 == test_monitor.episodes
	assert 50 == test_monitor.plot_interval
	assert os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + 'QLearningCEAgent_CircularEconomy.dat' == test_monitor.path_to_modelfile
	assert isinstance(test_monitor.marketplace, sim_market.CircularEconomy)
	assert isinstance(test_monitor.agents[0], agent.QLearningCEAgent)
	assert 1 == len(test_monitor.agents)
	assert ['#0000ff'] == test_monitor.agent_colors
	assert test_monitor.subfolder_name.startswith('plots_')
	assert os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) + os.sep + 'monitoring' + os.sep + test_monitor.subfolder_name == test_monitor.folder_path


def test_setup_with_invalid_agents():
	with pytest.raises(AssertionError):
		monitor.setup_monitoring(agents=[agent.FixedPriceLEAgent, agent.FixedPriceCERebuyAgent])


def test_setup_with_valid_agents():
	monitor.setup_monitoring(agents=[agent.FixedPriceCERebuyAgent, agent.FixedPriceCEAgent])


def test_metrics_average():
	assert 6 == monitor.metrics_average(create_mock_rewards())


def test_metrics_median():
	assert 6 == monitor.metrics_median(create_mock_rewards())


def test_metrics_maximum():
	assert 11 == monitor.metrics_maximum(create_mock_rewards())


def test_metrics_minimum():
	assert 1 == monitor.metrics_minimum(create_mock_rewards())


def test_round_up():
	assert monitor.round_up(999, -3) == 1000


# all arrays in rewards must be of the same size
def test_rewards_array_size():
	# Numpy doesn't like nested arrays of different sizes, need to specify dtype=object
	rewards_wrong = np.array([[1, 2], [1, 2, 3]], dtype=object)
	with pytest.raises(Exception):
		monitor.create_histogram(rewards_wrong)


def test_get_episode_reward():
	saved_episode_length = ut.EPISODE_LENGTH
	ut.EPISODE_LENGTH = 2
	all_steps_reward = [[1, 2, 3, 4], [4, 5, 6, 7], [1, 3, 4, 5]]
	assert [[3, 7], [9, 13], [4, 9]] == monitor.get_episode_rewards(all_steps_reward)
	ut.EPISODE_LENGTH = saved_episode_length


agent_rewards_histogram = [
	([agent.RuleBasedCEAgent], [[100, 0]]),
	([agent.RuleBasedCEAgent, agent.RuleBasedCEAgent], [[100, 0], [10, 5]]),
	([agent.RuleBasedCEAgent, agent.RuleBasedCEAgent, agent.RuleBasedCEAgent, agent.RuleBasedCEAgent],
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


def test_run_marketplace():
	monitor.setup_monitoring(episodes=100, plot_interval=100, agents=[agent.FixedPriceCEAgent])
	agent_rewards = monitor.run_marketplace()
	print(agent_rewards)
	assert 1 == len(monitor.agents)
	assert 100 * ut.EPISODE_LENGTH == len(agent_rewards[0])


def test_main():
	am.monitor.setup_monitoring(draw_enabled=False, episodes=10, plot_interval=10, subfolder_name='test_plots_')
	am.main()
	assert os.path.exists(am.monitor.folder_path)
