import pytest

from .context import FixedPriceAgent, RuleBasedCEAgent
from .context import utils as ut

test_state = [50, 60]


def test_fixed_price_agent_returns_default_fixed_price():
	test_agent = FixedPriceAgent()
	assert 42 == test_agent.policy(test_state)


def test_fixed_price_agent_returns_fixed_price():
	test_agent = FixedPriceAgent(35)
	assert 35 == test_agent.policy(test_state)


def test_helper_function_action_to_array():
	test_agent = RuleBasedCEAgent()
	assert [3, 4] == test_agent.action_to_array(34)


def test_helper_function_array_to_action():
	test_agent = RuleBasedCEAgent()
	assert 34 == test_agent.array_to_action([3, 4])


array_testing = [([20, 50], 68), ([40, 50], 57), ([60, 50], 46), ([80, 50], 29)]


@pytest.mark.parametrize('state, expected_prices', array_testing)
def test_storage_evaluation(state, expected_prices):
	# setting up test constants
	ut.MAX_STORAGE = 100
	ut.MAX_PRICE = 10
	ut.PRODUCTION_PRICE = 2
	test_agent = RuleBasedCEAgent()

	assert expected_prices == test_agent.policy(state)


def test_prices_are_not_higher_than_allowed():
	# setting up test constants
	ut.MAX_PRICE = 10
	ut.PRODUCTION_PRICE = 9

	test_agent = RuleBasedCEAgent()

	assert 99 >= test_agent.policy(test_state)
