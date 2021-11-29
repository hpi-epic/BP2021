import pytest


from .context import utils as ut
from .context import FixedPriceAgent
from .context import RuleBasedCEAgent

test_state = [50,60]

def test_fixed_price_agent_returns_default_fixed_price():
	test_agent = FixedPriceAgent()
	assert 42 == test_agent.policy(test_state)

def test_fixed_price_agent_returns_fixed_price():
	test_agent = FixedPriceAgent(35)
	assert 35 == test_agent.policy(test_state)

array_testing = [([20,50], 68), ([40,50], 57), ([60,50], 46), ([80,50], 29)]

@pytest.mark.parametrize('state, expected_prices', array_testing)
def test_storage_evaluation(state, expected_prices):
	# setting up test constants
	ut.MAX_STORAGE = 100
	ut.MAX_PRICE = 10
	ut.PRODUCTION_PRICE = 2
	test_agent = RuleBasedCEAgent()
	assert expected_prices == test_agent.policy(state)
