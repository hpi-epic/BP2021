from importlib import reload

from .context import agent_monitoring as am


# create mock rewards list
def create_mock_rewards() -> list:
	mock_rewards = []
	for number in range(1, 12):
		mock_rewards.append(number)
	return mock_rewards


def test_run_agent_monitoring():
	reload(am)


def test_metrics_average():
	assert 6 == am.metrics_average(create_mock_rewards())


def test_metrics_median():
	assert 6 == am.metrics_median(create_mock_rewards())


def test_metrics_maximum():
	assert 11 == am.metrics_maximum(create_mock_rewards())


def test_metrics_minimum():
	assert 1 == am.metrics_minimum(create_mock_rewards())


def test_round_up():
	assert am.round_up(999, -3) == 1000
