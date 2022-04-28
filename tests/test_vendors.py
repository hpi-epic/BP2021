import json
from importlib import reload
from unittest.mock import mock_open, patch

import numpy as np
import pytest
import utils_tests as ut_t
from numpy import random

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader, HyperparameterConfig
import recommerce.market.circular.circular_vendors as circular_vendors
import recommerce.market.linear.linear_vendors as linear_vendors
import recommerce.market.vendors as vendors
from recommerce.market.linear.linear_sim_market import LinearEconomyOligopoly
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


config_hyperparameter: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')

abstract_agent_classes_testcases = [
	vendors.Agent,
	circular_vendors.CircularAgent,
	linear_vendors.LinearAgent,
	vendors.HumanPlayer,
	vendors.RuleBasedAgent,
	vendors.FixedPriceAgent,
	ReinforcementLearningAgent
]


@pytest.mark.parametrize('agent', abstract_agent_classes_testcases)
def test_abstract_agent_classes(agent):
	with pytest.raises(TypeError) as error_message:
		agent()
	assert 'Can\'t instantiate abstract class' in str(error_message.value)


non_abstract_agent_classes_testcases = [
	linear_vendors.HumanPlayerLE,
	circular_vendors.HumanPlayerCE,
	circular_vendors.HumanPlayerCERebuy,
	circular_vendors.FixedPriceCEAgent,
	circular_vendors.FixedPriceCERebuyAgent,
	linear_vendors.FixedPriceLEAgent,
	circular_vendors.RuleBasedCEAgent,
	circular_vendors.RuleBasedCERebuyAgent,
	circular_vendors.RuleBasedCERebuyAgentCompetitive,
	circular_vendors.RuleBasedCERebuyAgentStorageMinimizer
]


@pytest.mark.parametrize('agent', non_abstract_agent_classes_testcases)
def test_non_abstract_agent_classes(agent):
	agent()


def test_non_abstract_qlearning_agent():
	QLearningAgent(marketplace=LinearEconomyOligopoly())


fixed_price_agent_observation_policy_pairs_testcases = [
	(linear_vendors.FixedPriceLEAgent(config=config_hyperparameter), config_hyperparameter.production_price + 3),
	(linear_vendors.FixedPriceLEAgent(config=config_hyperparameter, fixed_price=7), 7),
	(circular_vendors.FixedPriceCEAgent(), (2, 4)),
	(circular_vendors.FixedPriceCEAgent((3, 5)), (3, 5)),
	(circular_vendors.FixedPriceCERebuyAgent(), (3, 6, 2)),
	(circular_vendors.FixedPriceCERebuyAgent((4, 7, 3)), (4, 7, 3))
]


@pytest.mark.parametrize('agent, expected_result', fixed_price_agent_observation_policy_pairs_testcases)
def test_fixed_price_agent_observation_policy_pairs(agent, expected_result):
	# state doesn't actually matter since we test fixed price agents
	assert expected_result == agent.policy([50, 60])


storage_evaluation_testcases = [
	([50, 5], (6, 9)),
	([50, 9], (5, 8)),
	([50, 12], (4, 7)),
	([50, 15], (2, 9))
]


@pytest.mark.parametrize('state, expected_prices', storage_evaluation_testcases)
def test_storage_evaluation(state, expected_prices):
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(
		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(max_price=10, production_price=2)))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		agent = circular_vendors.RuleBasedCEAgent(config=config_hyperparameter)
		assert expected_prices == agent.policy(state)


storage_evaluation_with_rebuy_price_testcases = [
	([50, 5], (6, 9, 5)),
	([50, 9], (5, 8, 3)),
	([50, 12], (4, 7, 2)),
	([50, 15], (2, 9, 0))
]


@pytest.mark.parametrize('state, expected_prices', storage_evaluation_with_rebuy_price_testcases)
def test_storage_evaluation_with_rebuy_price(state, expected_prices):
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(
		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(max_price=10, production_price=2)))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		agent = circular_vendors.RuleBasedCERebuyAgent(config=config_hyperparameter)
		assert expected_prices == agent.policy(state)


def test_prices_are_not_higher_than_allowed():
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(
		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(max_price=10, production_price=9)))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		test_agent = circular_vendors.RuleBasedCEAgent(config=config_hyperparameter)
		assert (9, 9) >= test_agent.policy([50, 60])


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour.
# This is dependent on the sim_market working!
# TODO: Make deterministic #174
def random_offer_linear_duopoly():
	return [random.randint(1, config_hyperparameter.max_quality), random.randint(1, config_hyperparameter.max_price), random.randint(1, config_hyperparameter.max_quality)]


def random_offer_circular_oligopoly(is_rebuy_economy: bool):
	single_comp_prices = [
		random.randint(1, config_hyperparameter.max_price),
		random.randint(1, config_hyperparameter.max_price),
		random.randint(0, config_hyperparameter.max_storage)
		]
	viewed_agent_list = [random.randint(1, 1000), random.randint(0, config_hyperparameter.max_storage)]
	observation = viewed_agent_list
	if is_rebuy_economy:
		for _ in range(4):
			observation += [random.randint(1, config_hyperparameter.max_price)] + single_comp_prices
		return np.array(observation)
	else:
		for _ in range(4):
			observation += single_comp_prices
		return np.array(observation)

# TODO: figure out which rule-based agents perform their policies with "production_price-increment"
# policy_testcases = []


# # Test the policy()-function of the different competitors
# # TODO: Update this test for all current competitors
# @pytest.mark.parametrize('competitor_class, state', policy_testcases)
# def test_policy(competitor_class, state):
# 	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(
# 		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(max_price=10, production_price=2)))
# 	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
# 		ut_t.check_mock_file(mock_file, mock_json)
# 		import_config()
# 		competitor = competitor_class()

# 		assert config.production_price <= competitor.policy(state) < config.max_price


policy_plus_one_testcases = [
	(linear_vendors.CompetitorLinearRatio1, random_offer_linear_duopoly()),
	(linear_vendors.CompetitorRandom, random_offer_linear_duopoly()),
	(linear_vendors.CompetitorJust2Players, random_offer_linear_duopoly())
]


# Test the policy()-function of the different competitors.
# This test differs from the one before that these competitors use config.PRODUCTION_PRICE + 1
# TODO: Update this test for all current competitors
@pytest.mark.parametrize('competitor_class, state', policy_plus_one_testcases)
def test_policy_plus_one(competitor_class, state):
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(
		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(max_price=10, production_price=2)))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		competitor = competitor_class(config=config_hyperparameter)

		assert config_hyperparameter.production_price + 1 <= competitor.policy(state) < config_hyperparameter.max_price


clamp_price_testcases = [
	10,
	-1,
	5
]


@pytest.mark.parametrize('price', clamp_price_testcases)
def test_clamp_price(price):
	assert 0 <= circular_vendors.RuleBasedCEAgent()._clamp_price(price, 0, 9) <= 9


def test_get_competitors_prices_with_rebuy():
	observation = random_offer_circular_oligopoly(is_rebuy_economy=True)
	competitors_refurbished_prices, competitors_new_prices, competitors_rebuy_prices = \
		circular_vendors.RuleBasedCERebuyAgentCompetitive()._get_competitor_prices(observation=observation, is_rebuy_economy=True)
	assert len(competitors_new_prices) == len(competitors_rebuy_prices) == len(competitors_refurbished_prices)
	for competitor in range(len(competitors_new_prices)):
		assert competitors_refurbished_prices[competitor] == observation[(competitor * 4) + 2]
		assert competitors_new_prices[competitor] == observation[(competitor * 4) + 3]
		assert competitors_rebuy_prices[competitor] == observation[(competitor * 4) + 4]


def test_get_competitors_prices():
	observation = random_offer_circular_oligopoly(is_rebuy_economy=False)
	competitors_refurbished_prices, competitors_new_prices = \
		circular_vendors.RuleBasedCEAgent()._get_competitor_prices(observation=observation, is_rebuy_economy=False)
	assert len(competitors_new_prices) == len(competitors_refurbished_prices)
	for competitor in range(len(competitors_new_prices)):
		assert competitors_refurbished_prices[competitor] == observation[(competitor * 4) + 2]
		assert competitors_new_prices[competitor] == observation[(competitor * 4) + 3]
