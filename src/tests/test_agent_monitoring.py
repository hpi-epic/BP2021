from importlib import reload

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


def test_run_agent_monitoring():
	reload(am)


# TODO: add test for setup_monitoring
# TODO: should probably add a fixture setting up a Monitor instance
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
