import os
import re
import shutil

import numpy as np
import pytest

from .context import Monitor, agent
from .context import agent_monitoring as am
from .context import utils as ut

monitor = Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = Monitor()
	monitor.setup_monitoring(draw_enabled=False, subfolder_path='test_plots_')


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


# values and types are mismatched on purpose, as we just want to make sure the global values are changed correctly, we don't work with them
def test_setup_monitoring():
	monitor.setup_monitoring(0, 1, 2, 3, 4, 5, [6])
	assert 0 == monitor.enable_live_draws
	assert 1 == monitor.episodes
	assert 2 == monitor.plot_interval
	assert 3 == monitor.path_to_modelfile
	assert 4 == monitor.situation
	assert 5 == monitor.marketplace
	assert [6] == monitor.agents
	assert 1 == len(monitor.agent_colors)


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
	([agent.RuleBasedCEAgent()], [[100, 0]]),
	([agent.RuleBasedCEAgent(), agent.RuleBasedCEAgent()], [[100, 0], [10, 5]]),
	([agent.RuleBasedCEAgent(), agent.RuleBasedCEAgent(), agent.RuleBasedCEAgent(), agent.RuleBasedCEAgent()],
		[[100, 0], [10, 5], [100, 10000], [10, 1000]])
]


@pytest.mark.parametrize('agents, rewards', agent_rewards_histogram)
def test_create_histogram(agents, rewards):
	monitor.setup_monitoring(agents=agents)
	monitor.create_histogram(rewards)


@pytest.mark.parametrize('agents, rewards', agent_rewards_histogram)
def test_create_stat_plots(agents, rewards):
	monitor.setup_monitoring(agents=agents, episodes=len(rewards[0]), plot_interval=1)
	monitor.create_stat_plots(rewards)


def test_run_marketplace():
	monitor.setup_monitoring(episodes=100, plot_interval=100, agents=[agent.FixedPriceLEAgent(5)])
	print(monitor.run_marketplace())
	print('###########', monitor.agents)
	assert 1 == len(monitor.agents)
	assert 100 * ut.EPISODE_LENGTH == len(monitor.run_marketplace()[0])


def test_main():
	am.monitor.setup_monitoring(draw_enabled=False, episodes=10, plot_interval=10, subfolder_path='test_plots_')
	am.main()
	assert os.path.exists(am.monitor.folder_path)
