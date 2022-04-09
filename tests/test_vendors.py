from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t
from numpy import random

import recommerce.configuration.hyperparameter_config as hyperparameter_config
import recommerce.market.circular.circular_vendors as circular_vendors
import recommerce.market.linear.linear_vendors as linear_vendors
import recommerce.market.vendors as vendors
from recommerce.market.linear.linear_sim_market import MultiCompetitorScenario
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent, QLearningCEAgent, QLearningCERebuyAgent, QLearningLEAgent
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


def teardown_module(module):
	print('***TEARDOWN***')
	reload(hyperparameter_config)


def import_config() -> hyperparameter_config.HyperparameterConfig:
	"""
	Reload the hyperparameter_config file to update the config variable with the mocked values.

	Returns:
		HyperparameterConfig: The config object.
	"""
	reload(hyperparameter_config)
	return hyperparameter_config.config


config = import_config()

abstract_agent_classes_testcases = [
	vendors.Agent,
	circular_vendors.CircularAgent,
	linear_vendors.LinearAgent,
	vendors.HumanPlayer,
	vendors.RuleBasedAgent,
	vendors.FixedPriceAgent,
	ReinforcementLearningAgent,
	QLearningAgent
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
	circular_vendors.RuleBasedCERebuyAgent
]


@pytest.mark.parametrize('agent', non_abstract_agent_classes_testcases)
def test_non_abstract_agent_classes(agent):
	agent()


# actual n_observation and n_action are not needed, we just test if the initialization fails or not
non_abstract_qlearning_agent_classes_testcases = [
	(QLearningLEAgent, MultiCompetitorScenario),
	(QLearningCEAgent, MultiCompetitorScenario),
	(QLearningCERebuyAgent, MultiCompetitorScenario)
]


@pytest.mark.parametrize('agent, marketplace_class', non_abstract_qlearning_agent_classes_testcases)
def test_non_abstract_qlearning_agent_classes(agent, marketplace_class):
	agent(marketplace=marketplace_class())


fixed_price_agent_observation_policy_pairs_testcases = [
	(linear_vendors.FixedPriceLEAgent(), config.production_price + 3),
	(linear_vendors.FixedPriceLEAgent(7), 7),
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
	json = ut_t.create_hyperparameter_mock_json(
		sim_market=ut_t.create_hyperparameter_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		import_config()
		agent = circular_vendors.RuleBasedCEAgent()
		assert expected_prices == agent.policy(state)


storage_evaluation_with_rebuy_price_testcases = [
	([50, 5], (6, 9, 5)),
	([50, 9], (5, 8, 3)),
	([50, 12], (4, 7, 2)),
	([50, 15], (2, 9, 0))
]


@pytest.mark.parametrize('state, expected_prices', storage_evaluation_with_rebuy_price_testcases)
def test_storage_evaluation_with_rebuy_price(state, expected_prices):
	json = ut_t.create_hyperparameter_mock_json(
		sim_market=ut_t.create_hyperparameter_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		import_config()
		agent = circular_vendors.RuleBasedCERebuyAgent()
		assert expected_prices == agent.policy(state)


def test_prices_are_not_higher_than_allowed():
	json = ut_t.create_hyperparameter_mock_json(
		sim_market=ut_t.create_hyperparameter_mock_json_sim_market(max_price='10', production_price='9'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		import_config()
		test_agent = circular_vendors.RuleBasedCEAgent()
		assert (9, 9) >= test_agent.policy([50, 60])


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour.
# This is dependent on the sim_market working!
# TODO: Make deterministic #174
def random_offer():
	return [random.randint(1, config.max_quality), random.randint(1, config.max_price), random.randint(1, config.max_quality)]


policy_testcases = [
	(linear_vendors.CompetitorJust2Players, random_offer())
]


# Test the policy()-function of the different competitors
# TODO: Update this test for all current competitors
@pytest.mark.parametrize('competitor_class, state', policy_testcases)
def test_policy(competitor_class, state):
	json = ut_t.create_hyperparameter_mock_json(
		sim_market=ut_t.create_hyperparameter_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		import_config()
		competitor = competitor_class()

		assert config.production_price <= competitor.policy(state) < config.max_price


policy_plus_one_testcases = [
	(linear_vendors.CompetitorLinearRatio1, random_offer()),
	(linear_vendors.CompetitorRandom, random_offer())
]


# Test the policy()-function of the different competitors.
# This test differs from the one before that these competitors use config.PRODUCTION_PRICE + 1
# TODO: Update this test for all current competitors
@pytest.mark.parametrize('competitor_class, state', policy_plus_one_testcases)
def test_policy_plus_one(competitor_class, state):
	json = ut_t.create_hyperparameter_mock_json(
		sim_market=ut_t.create_hyperparameter_mock_json_sim_market(max_price='10', production_price='2'))
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		import_config()
		competitor = competitor_class()

		assert config.production_price + 1 <= competitor.policy(state) < config.max_price
