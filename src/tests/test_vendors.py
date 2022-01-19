from importlib import reload
from unittest.mock import mock_open, patch

import pytest
from numpy import random

import agents.vendors as vendors
import configuration.config as config
import tests.utils_tests as ut_t

abstract_agent_classes_testcases = [
	vendors.Agent,
	vendors.CircularAgent,
	vendors.LinearAgent,
	vendors.HumanPlayer,
	vendors.FixedPriceAgent
]


@pytest.mark.parametrize('agent', abstract_agent_classes_testcases)
def test_abstract_agent_classes(agent):
	with pytest.raises(TypeError):
		agent()


non_abstract_agent_classes_testcases = [
	vendors.HumanPlayerLE,
	vendors.HumanPlayerCE,
	vendors.HumanPlayerCERebuy,
	vendors.FixedPriceCEAgent,
	vendors.FixedPriceCERebuyAgent,
	vendors.FixedPriceLEAgent,
	vendors.RuleBasedCEAgent,
	vendors.RuleBasedCERebuyAgent
]


@pytest.mark.parametrize('agent', non_abstract_agent_classes_testcases)
def test_non_abstract_agent_classes(agent):
	agent()


# actual n_observation and n_action are not needed, we just test if the initialization fails or not
non_abstract_qlearning_agent_classes_testcases = [
	(vendors.QLearningAgent, 10, 10),
	(vendors.QLearningCEAgent, 10, 10),
	(vendors.QLearningCERebuyAgent, 10, 10)
]


@pytest.mark.parametrize('agent, n_observation, n_actions', non_abstract_qlearning_agent_classes_testcases)
def test_non_abstract_qlearning_agent_classes(agent, n_observation, n_actions):
	agent(n_observation, n_actions)


fixed_price_agent_observation_policy_pairs_testcases = [
	(vendors.FixedPriceLEAgent(), config.PRODUCTION_PRICE + 3),
	(vendors.FixedPriceLEAgent(7), 7),
	(vendors.FixedPriceCEAgent(), (2, 4)),
	(vendors.FixedPriceCEAgent((3, 5)), (3, 5)),
	(vendors.FixedPriceCERebuyAgent(), (3, 6, 2)),
	(vendors.FixedPriceCERebuyAgent((4, 7, 3)), (4, 7, 3))
]


@pytest.mark.parametrize('agent, expected_result', fixed_price_agent_observation_policy_pairs_testcases)
def test_fixed_price_agent_observation_policy_pairs(agent, expected_result):
	# state doesn't actually matter since we test fixed price agents
	assert expected_result == agent.policy([50, 60])


storage_evaluation_testcases = [
	([50, 5], (6, 8)),
	([50, 9], (5, 7)),
	([50, 12], (4, 6)),
	([50, 15], (2, 9))
]


@pytest.mark.parametrize('state, expected_prices', storage_evaluation_testcases)
def test_storage_evaluation(state, expected_prices):
	json = ut_t.create_mock_json(sim_market=ut_t.create_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		reload(config)
		agent = vendors.RuleBasedCEAgent()
		assert expected_prices == agent.policy(state)


storage_evaluation_with_rebuy_price_testcases = [
	([50, 5], (6, 8, 5)),
	([50, 9], (5, 7, 3)),
	([50, 12], (4, 6, 2)),
	([50, 15], (2, 9, 0))
]


@pytest.mark.parametrize('state, expected_prices', storage_evaluation_with_rebuy_price_testcases)
def test_storage_evaluation_with_rebuy_price(state, expected_prices):
	json = ut_t.create_mock_json(sim_market=ut_t.create_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		reload(config)
		agent = vendors.RuleBasedCERebuyAgent()
		assert expected_prices == agent.policy(state)


def test_prices_are_not_higher_than_allowed():
	json = ut_t.create_mock_json(sim_market=ut_t.create_mock_json_sim_market(max_price='10', production_price='9'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		reload(config)
		test_agent = vendors.RuleBasedCEAgent()
		assert (9, 9) >= test_agent.policy([50, 60])


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour.
# This is dependent on the sim_market working!
# TODO: Make deterministic #174
def random_offer():
	return [random.randint(1, config.MAX_QUALITY), random.randint(1, config.MAX_PRICE), random.randint(1, config.MAX_QUALITY)]


policy_testcases = [
	(vendors.CompetitorJust2Players, random_offer())
]


# Test the policy()-function of the different competitors
# TODO: Update this test for all current competitors
@pytest.mark.parametrize('competitor_class, state', policy_testcases)
def test_policy(competitor_class, state):
	json = ut_t.create_mock_json(sim_market=ut_t.create_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		reload(config)
		competitor = competitor_class()

		assert config.PRODUCTION_PRICE <= competitor.policy(state) < config.MAX_PRICE


policy_plus_one_testcases = [
	(vendors.CompetitorLinearRatio1, random_offer()),
	(vendors.CompetitorRandom, random_offer())
]


# Test the policy()-function of the different competitors.
# This test differs from the one before that these competitors use config.PRODUCTION_PRICE + 1
# TODO: Update this test for all current competitors
@pytest.mark.parametrize('competitor_class, state', policy_plus_one_testcases)
def test_policy_plus_one(competitor_class, state):
	json = ut_t.create_mock_json(sim_market=ut_t.create_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		reload(config)
		competitor = competitor_class()

		assert config.PRODUCTION_PRICE + 1 <= competitor.policy(state) < config.MAX_PRICE
