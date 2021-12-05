import os
import re

from .context import Monitor
from .context import agent_monitoring as am

monitor = None


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = Monitor()


# create mock rewards list
def create_mock_rewards() -> list:
	mock_rewards = []
	for number in range(1, 12):
		mock_rewards.append(number)
	return mock_rewards


# new_episodes, new_interval, new_modelfile, new_situation, new_marketplace, new_agent
# values and types are mismatched on purpose, as we just want to make sure the global values are changed correctly, we don't work with them
def test_setup_monitoring():
	monitor.setup_monitoring(1, 2, 3, 4, 5, 6)
	assert 1 == monitor.episodes
	assert 2 == monitor.histogram_plot_interval
	assert 3 == monitor.path_to_modelfile
	assert 4 == monitor.situation
	assert 5 == monitor.marketplace
	assert 6 == monitor.agent


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


def test_create_histogram():
	rewards = [100, 1000, 0, 1538]
	monitor.create_histogram(rewards)


def test_reset_episode():
	reward_per_episode = 20
	is_done = True
	monitor.marketplace.state = []

	reward_per_episode, is_done, monitor.marketplace.state = monitor.reset_episode(reward_per_episode, is_done, monitor.marketplace)

	assert reward_per_episode == 0
	assert is_done is False
	assert len(monitor.marketplace.state) != 0


def test_run_marketplace():
	monitor.setup_monitoring(new_episodes=100, new_interval=100)
	assert len(monitor.run_marketplace()) == 100


def test_main():
	am.monitor.setup_monitoring(new_episodes=10, new_interval=10)
	am.main()
	# make sure a final_plot was saved, and remove it after the test
	for f in os.listdir('./monitoring'):
		if re.match('final_plot*', f):
			os.remove('./monitoring/' + f)
			return True
	assert False
