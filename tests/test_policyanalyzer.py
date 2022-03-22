import os

import numpy as np
import pytest

from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent
from recommerce.market.linear.linear_vendors import CompetitorLinearRatio1
from recommerce.monitoring.policyanalyzer import PolicyAnalyzer
from recommerce.rl.actorcritic.actorcritic_agent import ContinuosActorCriticAgentFixedOneStd
from recommerce.rl.q_learning.q_learning_agent import QLearningCERebuyAgent

# The load path for the agent modelos.paths
parameters_path = os.path.join('tests', 'test_data')
write_to_path = os.path.join('tests', 'test_results', 'policyanalyzer')


def test_rule_based_linear_competitor1():
	pa = PolicyAnalyzer(CompetitorLinearRatio1())
	pa.analyze_policy(np.array([15, -1, 10]), [(1, 'competitor price', range(1, 11))])
	assert os.path.exists(os.path.join(write_to_path, 'add_a_title_here.png'))


def test_rule_based_linear_competitor2():
	pa = PolicyAnalyzer(CompetitorLinearRatio1())
	pa.analyze_policy(np.array([-1, -1, 10]), [(0, 'own quality', range(5, 20)), (1, 'competitor price', range(1, 11))], "agent's policy")
	assert os.path.exists(os.path.join(write_to_path, 'agent\'s_policy.png'))


def test_rule_based_circular_competitor1():
	pa = PolicyAnalyzer(RuleBasedCERebuyAgent())
	pa.analyze_policy(np.array([50, -1]), [(1, 'self storage stock', range(0, 30))], 'refurbished price only on storage', 0)
	assert os.path.exists(os.path.join(write_to_path, 'refurbished_price_only_on_storage.png'))


def test_rule_based_circular_competitor2():
	pa = PolicyAnalyzer(RuleBasedCERebuyAgent())
	pa.analyze_policy(
		np.array([-1, -1]),
		[(1, 'self storage stock', range(0, 30)),
		(0, 'in circulation', range(0, 200))],
		'refurbished price', 0
	)
	assert os.path.exists(os.path.join(write_to_path, 'refurbished_price.png'))


monopoly_test_cases = [
	('refurbished price', 0, 'refurbished_price.png'),
	('new price', 1, 'new_price.png'),
	('rebuy price', 2, 'rebuy_price.png')
]


@pytest.mark.parametrize('title, policyaccess, expected_filename', monopoly_test_cases)
def test_circular_monopoly_q_learning(title, policyaccess, expected_filename):
	q_learing_agent = QLearningCERebuyAgent(
		2, 1000, load_path=os.path.join(parameters_path,
		'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat')
	)
	pa = PolicyAnalyzer(q_learing_agent)
	pa.analyze_policy(
		np.array([-1, -1]),
		[(1, 'self storage stock', range(0, 30)),
		(0, 'in circulation', range(0, 200))],
		title, policyaccess
	)
	assert os.path.exists(os.path.join(write_to_path, expected_filename))


one_competitor_test_cases = [
	('own refurbished price', 0, 'own_refurbished_price.png'),
	('own new price', 1, 'own_new_price.png'),
	('own rebuy price', 2, 'own_rebuy_price.png')
]


@pytest.mark.parametrize('title, policyaccess, expected_filename', one_competitor_test_cases)
def test_circular_duopol_continuos_actorcritic(title, policyaccess, expected_filename):
	q_learing_agent = ContinuosActorCriticAgentFixedOneStd(
		6, 3, load_path=os.path.join(parameters_path,
			'actor_parametersCircularEconomyRebuyPriceOneCompetitor_ContinuosActorCriticAgentFixedOneStd.dat')
	)
	pa = PolicyAnalyzer(q_learing_agent)
	pa.analyze_policy(
		np.array([75, 10, -1, -1, 2, 12]),
		[(1, "competitor's refurbished price", range(0, 10)),
		(0, "competitor's new price", range(0, 10))],
		title, policyaccess
	)
	assert os.path.exists(os.path.join(write_to_path, expected_filename))
