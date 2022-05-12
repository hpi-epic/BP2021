import os
import time

import numpy as np
import pytest
import utils_tests as ut_t

from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly, CircularEconomyRebuyPriceMonopoly
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent, RuleBasedCERebuyAgentCompetitive
from recommerce.market.linear.linear_vendors import LinearRatio1LEAgent
from recommerce.monitoring.policyanalyzer import PolicyAnalyzer
from recommerce.rl.actorcritic.actorcritic_agent import ContinuousActorCriticAgentFixedOneStd
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

write_to_path = os.path.join(PathManager.results_path, 'policyanalyzer')

config_hyperparameter: HyperparameterConfig = ut_t.mock_config_hyperparameter()


def test_rule_based_linear_competitor1():
	pa = PolicyAnalyzer(LinearRatio1LEAgent(config=config_hyperparameter))
	given_path = pa.analyze_policy(np.array([15, -1, 10]), [(1, 'competitor price', range(1, 11))])
	expected_path = os.path.join(write_to_path, 'add_a_title_here.png')
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)


def test_rule_based_linear_competitor2():
	pa = PolicyAnalyzer(LinearRatio1LEAgent(config=config_hyperparameter))
	given_path = pa.analyze_policy(
		np.array([-1, -1, 10]),
		[(0, 'own quality', range(5, 20)), (1, 'competitor price', range(1, 11))],
		"agent's policy"
	)
	expected_path = os.path.join(write_to_path, "agent's_policy.png")
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)


def test_rule_based_circular_competitor1():
	pa = PolicyAnalyzer(RuleBasedCERebuyAgent(config=config_hyperparameter))
	given_path = pa.analyze_policy(np.array([50, -1]), [(1, 'self storage stock', range(30))], 'refurbished price only on storage', 0)
	expected_path = os.path.join(write_to_path, 'refurbished_price_only_on_storage.png')
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)


def test_rule_based_circular_competitor2():
	pa = PolicyAnalyzer(RuleBasedCERebuyAgent(config=config_hyperparameter))
	given_path = pa.analyze_policy(
		np.array([-1, -1]),
		[(1, 'self storage stock', range(30)), (0, 'in circulation', range(200))],
		'refurbished price', 0
	)
	expected_path = os.path.join(write_to_path, 'refurbished_price.png')
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)


monopoly_test_cases = [
	('refurbished price', 0, 'refurbished_price.png'),
	('new price', 1, 'new_price.png'),
	('rebuy price', 2, 'rebuy_price.png')
]


@pytest.mark.parametrize('title, policyaccess, expected_filename', monopoly_test_cases)
def test_circular_monopoly_q_learning(title, policyaccess, expected_filename):
	q_learing_agent = QLearningAgent(
		marketplace=CircularEconomyRebuyPriceMonopoly(config=config_hyperparameter),
		config=config_hyperparameter,
		load_path=os.path.join(PathManager.data_path, 'CircularEconomyRebuyPriceMonopoly_QLearningAgent.dat')
	)
	pa = PolicyAnalyzer(q_learing_agent)
	given_path = pa.analyze_policy(
		np.array([-1, -1]),
		[(1, 'self storage stock', range(30)), (0, 'in circulation', range(200))],
		title, policyaccess
	)
	expected_path = os.path.join(write_to_path, expected_filename)
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)


one_competitor_test_cases = [
	('ql own refurbished price', 0, 'ql_own_refurbished_price.png'),
	('ql own new price', 1, 'ql_own_new_price.png'),
	('ql own rebuy price', 2, 'ql_own_rebuy_price.png')
]


@pytest.mark.parametrize('title, policyaccess, expected_filename', one_competitor_test_cases)
def test_circular_duopol_q_learning(title, policyaccess, expected_filename):
	q_learing_agent = QLearningAgent(
		marketplace=CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter),
		config=config_hyperparameter,
		load_path=os.path.join(PathManager.data_path, 'CircularEconomyRebuyPriceDuopoly_QLearningAgent.dat')
	)
	pa = PolicyAnalyzer(q_learing_agent)
	given_path = pa.analyze_policy(
		np.array([75, 10, -1, -1, 2, 12]),
		[(2, "competitor's refurbished price", range(10)), (3, "competitor's new price", range(10))],
		title, policyaccess
	)
	expected_path = os.path.join(write_to_path, expected_filename)
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)


one_competitor_test_cases = [
	('a2c own refurbished price', 0, 'a2c_own_refurbished_price.png'),
	('a2c own new price', 1, 'a2c_own_new_price.png'),
	('a2c own rebuy price', 2, 'a2c_own_rebuy_price.png')
]


@pytest.mark.parametrize('title, policyaccess, expected_filename', one_competitor_test_cases)
def test_circular_duopol_continuous_actorcritic(title, policyaccess, expected_filename):
	a2c_agent = ContinuousActorCriticAgentFixedOneStd(
		marketplace=CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter),
		config=config_hyperparameter,
		load_path=os.path.join(PathManager.data_path,
			'actor_parametersCircularEconomyRebuyPriceDuopoly_ContinuousActorCriticAgentFixedOneStd.dat')
	)
	pa = PolicyAnalyzer(a2c_agent)
	given_path = pa.analyze_policy(
		np.array([75, 10, -1, -1, 2, 12]),
		[(2, "competitor's refurbished price", range(10)), (3, "competitor's new price", range(10))],
		title, policyaccess
	)
	expected_path = os.path.join(write_to_path, expected_filename)
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)


one_competitor_test_cases = [
	('rule based own refurbished price', 0, 'rule_based_own_refurbished_price.png'),
	('rule based own new price', 1, 'rule_based_own_new_price.png'),
	('rule based own rebuy price', 2, 'rule_based_own_rebuy_price.png')
]


@pytest.mark.parametrize('title, policyaccess, expected_filename', one_competitor_test_cases)
def test_circular_duopol_rule_based_agent(title, policyaccess, expected_filename):
	pa = PolicyAnalyzer(RuleBasedCERebuyAgentCompetitive(config=config_hyperparameter), 'rule_based_competitive_policy')
	given_path = pa.analyze_policy(
		np.array([75, 10, -1, -1, 2, 12]),
		[(2, "competitor's refurbished price", range(10)), (3, "competitor's new price", range(10))],
		title, policyaccess
	)
	expected_path = os.path.join(write_to_path, 'rule_based_competitive_policy', expected_filename)
	assert expected_path in given_path
	time.sleep(0.001)
	assert os.path.exists(expected_path)
