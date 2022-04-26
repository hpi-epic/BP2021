import os
from unittest.mock import patch

import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.circular.circular_vendors as circular_vendors
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.market.linear.linear_vendors import FixedPriceLEAgent
from recommerce.monitoring.exampleprinter import ExamplePrinter
from recommerce.rl.actorcritic.actorcritic_agent import ContinuosActorCriticAgentFixedOneStd, DiscreteActorCriticAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

# The load path for the agent modelfiles
parameters_path = os.path.join('tests', 'test_data')


def test_setup_exampleprinter():
	printer = ExamplePrinter()
	printer.setup_exampleprinter(marketplace=linear_market.LinearEconomyDuopoly(), agent=FixedPriceLEAgent())
	assert isinstance(printer.marketplace, linear_market.LinearEconomyDuopoly)
	assert isinstance(printer.agent, FixedPriceLEAgent)


full_episode_testcases_rule_based = [
	(linear_market.LinearEconomyDuopoly(), FixedPriceLEAgent()),
	(linear_market.LinearEconomyOligopoly(), FixedPriceLEAgent()),
	(circular_market.CircularEconomyMonopoly(), circular_vendors.FixedPriceCEAgent()),
	(circular_market.CircularEconomyMonopoly(), circular_vendors.RuleBasedCEAgent()),
	(circular_market.CircularEconomyRebuyPriceMonopoly(), circular_vendors.FixedPriceCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceMonopoly(), circular_vendors.RuleBasedCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceDuopoly(), circular_vendors.FixedPriceCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceDuopoly(), circular_vendors.RuleBasedCERebuyAgent())
]


@pytest.mark.parametrize('marketplace, agent', full_episode_testcases_rule_based)
def test_full_episode_rule_based(marketplace, agent):
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'),\
		patch('recommerce.monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter()
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example() >= -5000


full_episode_testcases_rl_agent = [
	(linear_market.LinearEconomyDuopoly(), QLearningAgent, 'LinearEconomyDuopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyMonopoly(), QLearningAgent,
		'CircularEconomyMonopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceMonopoly(), QLearningAgent,
		'CircularEconomyRebuyPriceMonopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(), QLearningAgent,
		'CircularEconomyRebuyPriceDuopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(), ContinuosActorCriticAgentFixedOneStd,
		'actor_parametersCircularEconomyRebuyPriceDuopoly_ContinuosActorCriticAgentFixedOneStd.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(), DiscreteActorCriticAgent,
		'actor_parametersCircularEconomyRebuyPriceDuopoly_DiscreteACACircularEconomyRebuy.dat')
]


@pytest.mark.parametrize('marketplace, agent_class, parameters_file', full_episode_testcases_rl_agent)
def test_full_episode_rl_agents(marketplace, agent_class, parameters_file):
	agent = agent_class(marketplace=marketplace, load_path=os.path.join(parameters_path, parameters_file))
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'),\
		patch('recommerce.monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter()
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example() >= -5000


@pytest.mark.slow
def test_exampleprinter_with_tensorboard():
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'):
		assert ExamplePrinter().run_example() >= -5000
