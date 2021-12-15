import pytest

from .context import agent
from .context import utils as ut


def test_abstract_agent_classes():
	with pytest.raises(TypeError):
		agent.Agent()
	with pytest.raises(TypeError):
		agent.CircularAgent()
	with pytest.raises(TypeError):
		agent.LinearAgent()
	with pytest.raises(TypeError):
		agent.HumanPlayer()
	with pytest.raises(TypeError):
		agent.FixedPriceAgent()
	# The QLearningAgent should be an abstract class, but since all of its child classes use the same methods it is not actually abstract
	# with pytest.raises(TypeError):
	# 	agent.QLearningAgent(10, 10)


def test_not_abstract_agent_classes():
	agent.HumanPlayerLE()
	agent.HumanPlayerCE()
	agent.HumanPlayerCERebuy()
	agent.FixedPriceCEAgent()
	agent.FixedPriceCERebuyAgent()
	agent.FixedPriceLEAgent()
	agent.RuleBasedCEAgent()
	agent.RuleBasedCERebuyAgent()
	agent.QLearningCEAgent(10, 10)
	agent.QLearningCERebuyAgent(10, 10)


test_state = [50, 60]


def test_fixed_price_LE_agent_returns_default_fixed_price():
	test_agent = agent.FixedPriceLEAgent()
	assert ut.PRODUCTION_PRICE + 3 == test_agent.policy(test_state)


def test_fixed_price_LE_agent_returns_fixed_price():
	test_agent = agent.FixedPriceLEAgent(7)
	assert 7 == test_agent.policy(test_state)


def test_fixed_price_CE_agent_returns_default_fixed_price():
	test_agent = agent.FixedPriceCEAgent()
	assert (2, 4) == test_agent.policy(test_state)


def test_fixed_price_CE_agent_returns_fixed_price():
	test_agent = agent.FixedPriceCEAgent((3, 5))
	assert (3, 5) == test_agent.policy(test_state)


def test_fixed_price_CERebuy_agent_returns_default_fixed_price():
	test_agent = agent.FixedPriceCERebuyAgent()
	assert (3, 6, 2) == test_agent.policy(test_state)


def test_fixed_price_CERebuy_agent_returns_fixed_price():
	test_agent = agent.FixedPriceCERebuyAgent((4, 7, 3))
	assert (4, 7, 3) == test_agent.policy(test_state)


array_storage_evaluation = [([8, 50], (6, 8)), ([17, 50], (5, 7)), ([27, 50], (4, 6)), ([80, 50], (2, 9))]
@pytest.mark.parametrize('state, expected_prices', array_storage_evaluation)
def test_storage_evaluation(state, expected_prices):
	# setting up test constants
	ut.MAX_STORAGE = 100
	ut.MAX_PRICE = 10
	ut.PRODUCTION_PRICE = 2
	test_agent = agent.RuleBasedCEAgent()

	assert expected_prices == test_agent.policy(state)


array_testing_rebuy = [([8, 50], (6, 8, 5)), ([17, 50], (5, 7, 3)), ([27, 50], (4, 6, 2)), ([80, 50], (2, 9, 0))]
@pytest.mark.parametrize('state, expected_prices', array_testing_rebuy)
def test_storage_evaluation_with_rebuy_price(state, expected_prices):
	# setting up test constants
	ut.MAX_STORAGE = 100
	ut.MAX_PRICE = 10
	ut.PRODUCTION_PRICE = 2
	test_agent = agent.RuleBasedCERebuyAgent()

	assert expected_prices == test_agent.policy(state)


def test_prices_are_not_higher_than_allowed():
	# setting up test constants
	ut.MAX_PRICE = 10
	ut.PRODUCTION_PRICE = 9

	test_agent = agent.RuleBasedCEAgent()

	assert (9, 9) >= test_agent.policy(test_state)
