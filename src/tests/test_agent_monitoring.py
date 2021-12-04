from importlib import reload

from .context import Monitor as monitor
from .context import agent_monitoring as am


# create mock rewards list
def create_mock_rewards() -> list:
	mock_rewards = []
	for number in range(1, 12):
		mock_rewards.append(number)
	return mock_rewards


def test_run_agent_monitoring():
	reload(am)


# TODO: should probably add a fixture setting up a Monitor instance
def test_metrics_average():
	assert 6 == monitor.metrics_average(monitor, create_mock_rewards())


def test_metrics_median():
	assert 6 == monitor.metrics_median(monitor, create_mock_rewards())


def test_metrics_maximum():
	assert 11 == monitor.metrics_maximum(monitor, create_mock_rewards())


def test_metrics_minimum():
	assert 1 == monitor.metrics_minimum(monitor, create_mock_rewards())


def test_round_up():
	assert monitor.round_up(monitor, 999, -3) == 1000
