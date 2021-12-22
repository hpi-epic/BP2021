from importlib import reload
from unittest.mock import mock_open, patch

import pytest

import agent
import tests.utils_tests as ut_t
import utils_sim_market as ut


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


def test_not_abstract_agent_classes():
	agent.HumanPlayerLE()
	agent.HumanPlayerCE()
	agent.HumanPlayerCERebuy()
	agent.FixedPriceCEAgent()
	agent.FixedPriceCERebuyAgent()
	agent.FixedPriceLEAgent()
	agent.RuleBasedCEAgent()
	agent.RuleBasedCERebuyAgent()
	agent.QLearningAgent(10, 10)
	agent.QLearningCEAgent(10, 10)
	agent.QLearningCERebuyAgent(10, 10)


test_state = [50, 60]
fixed_price_testcases = [(agent.FixedPriceLEAgent(), ut.PRODUCTION_PRICE + 3), (agent.FixedPriceLEAgent(7), 7), (agent.FixedPriceCEAgent(), (2, 4)), (agent.FixedPriceCEAgent((3, 5)), (3, 5)), (agent.FixedPriceCERebuyAgent(), (3, 6, 2)), (agent.FixedPriceCERebuyAgent((4, 7, 3)), (4, 7, 3))]


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
		test_agent = agent.RuleBasedCEAgent()
		assert expected_prices == test_agent.policy(state)


array_testing_rebuy = [([50, 5], (6, 8, 5)), ([50, 9], (5, 7, 3)), ([50, 12], (4, 6, 2)), ([50, 15], (2, 9, 0))]


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
