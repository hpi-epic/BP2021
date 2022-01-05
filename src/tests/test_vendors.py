from importlib import reload
from unittest.mock import mock_open, patch

import pytest
from numpy import random

import tests.utils_tests as ut_t
import utils_sim_market as ut
import vendors
from vendors import CompetitorJust2Players as C2Players
from vendors import CompetitorLinearRatio1 as CLinear1
from vendors import CompetitorRandom as CRandom


def test_abstract_agent_classes():
	with pytest.raises(TypeError):
		vendors.Agent()
	with pytest.raises(TypeError):
		vendors.CircularAgent()
	with pytest.raises(TypeError):
		vendors.LinearAgent()
	with pytest.raises(TypeError):
		vendors.HumanPlayer()
	with pytest.raises(TypeError):
		vendors.FixedPriceAgent()


def test_not_abstract_agent_classes():
	vendors.HumanPlayerLE()
	vendors.HumanPlayerCE()
	vendors.HumanPlayerCERebuy()
	vendors.FixedPriceCEAgent()
	vendors.FixedPriceCERebuyAgent()
	vendors.FixedPriceLEAgent()
	vendors.RuleBasedCEAgent()
	vendors.RuleBasedCERebuyAgent()
	vendors.QLearningAgent(10, 10)
	vendors.QLearningCEAgent(10, 10)
	vendors.QLearningCERebuyAgent(10, 10)


test_state = [50, 60]
fixed_price_testcases = [(vendors.FixedPriceLEAgent(), ut.PRODUCTION_PRICE + 3), (vendors.FixedPriceLEAgent(7), 7), (vendors.FixedPriceCEAgent(), (2, 4)), (vendors.FixedPriceCEAgent((3, 5)), (3, 5)), (vendors.FixedPriceCERebuyAgent(), (3, 6, 2)), (vendors.FixedPriceCERebuyAgent((4, 7, 3)), (4, 7, 3))]


@pytest.mark.parametrize('test_agent, expected_result', fixed_price_testcases)
def test_agent_observation_policy_pairs(test_agent, expected_result):
	assert expected_result == test_agent.policy(test_state)


array_storage_evaluation = [([50, 5], (6, 8)), ([50, 9], (5, 7)), ([50, 12], (4, 6)), ([50, 15], (2, 9))]


@pytest.mark.parametrize('state, expected_prices', array_storage_evaluation)
def test_storage_evaluation(state, expected_prices):
	json = ut_t.create_mock_json_sim_market(max_price='10', production_price='2')
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)
		reload(ut)
		test_agent = vendors.RuleBasedCEAgent()
		assert expected_prices == test_agent.policy(state)


array_testing_rebuy = [([50, 5], (6, 8, 5)), ([50, 9], (5, 7, 3)), ([50, 12], (4, 6, 2)), ([50, 15], (2, 9, 0))]


@pytest.mark.parametrize('state, expected_prices', array_testing_rebuy)
def test_storage_evaluation_with_rebuy_price(state, expected_prices):
	json = ut_t.create_mock_json_sim_market(max_price='10', production_price='2')
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)
		reload(ut)
		test_agent = vendors.RuleBasedCERebuyAgent()
		assert expected_prices == test_agent.policy(state)


def test_prices_are_not_higher_than_allowed():
	json = ut_t.create_mock_json_sim_market(max_price='10', production_price='9')
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)
		reload(ut)
		test_agent = vendors.RuleBasedCEAgent()
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
