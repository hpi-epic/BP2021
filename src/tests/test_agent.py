from importlib import reload
from unittest.mock import mock_open, patch

import pytest
from numpy import random

from .context import CompetitorJust2Players as C2Players
from .context import CompetitorLinearRatio1 as CLinear1
from .context import CompetitorRandom as CRandom
from .context import agent
from .context import utils_sim_market as ut
from .context import utils_tests as ut_t


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
	json = ut_t.create_mock_json_sim_market(max_price='10', production_price='2')
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)
		reload(ut)
		test_agent = agent.RuleBasedCEAgent()
		assert expected_prices == test_agent.policy(state)


array_testing_rebuy = [([8, 50], (6, 8, 5)), ([17, 50], (5, 7, 3)), ([27, 50], (4, 6, 2)), ([80, 50], (2, 9, 0))]


@pytest.mark.parametrize('state, expected_prices', array_testing_rebuy)
def test_storage_evaluation_with_rebuy_price(state, expected_prices):
	json = ut_t.create_mock_json_sim_market(max_price='10', production_price='2')
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)
		reload(ut)
		test_agent = agent.RuleBasedCERebuyAgent()
		assert expected_prices == test_agent.policy(state)


def test_prices_are_not_higher_than_allowed():
	json = ut_t.create_mock_json_sim_market(max_price='10', production_price='9')
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)
		reload(ut)
		test_agent = agent.RuleBasedCEAgent()
		assert (9, 9) >= test_agent.policy(test_state)


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer():
	return [random.randint(1, ut.MAX_QUALITY), random.randint(1, ut.MAX_PRICE), random.randint(1, ut.MAX_QUALITY)]


def get_competitor_pricing_ids():
	return [
		'Linear1', 'Random', '2Players'
	]


array_competitor_pricing = [
	(CLinear1, random_offer()),
	(CRandom, random_offer()),
	(C2Players, random_offer())
]


# Test the policy()-function of the different competitors
@pytest.mark.parametrize('competitor_class, state', array_competitor_pricing, ids=get_competitor_pricing_ids())
def test_policy(competitor_class, state):
	reload(ut)
	competitor = competitor_class()
	assert ut.PRODUCTION_PRICE == 2
	if competitor is CLinear1:
		assert ut.PRODUCTION_PRICE + 1 <= competitor.policy(competitor, state) < ut.MAX_PRICE
	if competitor is CRandom:
		assert ut.PRODUCTION_PRICE + 1 <= competitor.policy(competitor, state) < ut.MAX_PRICE
	if competitor is C2Players:
		assert ut.PRODUCTION_PRICE <= competitor.policy(competitor, state) < ut.MAX_PRICE
