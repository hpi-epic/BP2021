import os
from unittest.mock import patch

import pytest
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.circular.circular_vendors as circular_vendors
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.linear.linear_vendors import FixedPriceLEAgent
from recommerce.monitoring.exampleprinter import ExamplePrinter
from recommerce.rl.actorcritic.actorcritic_agent import ContinuousActorCriticAgentFixedOneStd, DiscreteActorCriticAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

# The load path for the agent modelfiles
parameters_path = os.path.join('tests', 'test_data')

config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config')


def test_setup_exampleprinter():
	printer = ExamplePrinter(config_market=config_market)
	printer.setup_exampleprinter(
		marketplace=linear_market.LinearEconomyDuopoly(config=config_market),
		agent=FixedPriceLEAgent(config=config_market)
		)
	assert isinstance(printer.marketplace, linear_market.LinearEconomyDuopoly)
	assert isinstance(printer.agent, FixedPriceLEAgent)


full_episode_testcases_rule_based = [
	(linear_market.LinearEconomyDuopoly(config=config_market),
		FixedPriceLEAgent(config=config_market)),
	(linear_market.LinearEconomyOligopoly(config=config_market),
		FixedPriceLEAgent(config=config_market)),
	(circular_market.CircularEconomyMonopoly(config=config_market),
		circular_vendors.FixedPriceCEAgent(config=config_market)),
	(circular_market.CircularEconomyMonopoly(config=config_market),
		circular_vendors.RuleBasedCEAgent(config=config_market)),
	(circular_market.CircularEconomyRebuyPriceMonopoly(config=config_market),
		circular_vendors.FixedPriceCERebuyAgent(config=config_market)),
	(circular_market.CircularEconomyRebuyPriceMonopoly(config=config_market),
		circular_vendors.RuleBasedCERebuyAgent(config=config_market)),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_market),
		circular_vendors.FixedPriceCERebuyAgent(config=config_market)),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_market),
		circular_vendors.RuleBasedCERebuyAgent(config=config_market))
]


@pytest.mark.parametrize('marketplace, agent', full_episode_testcases_rule_based)
def test_full_episode_rule_based(marketplace, agent):
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'),\
		patch('recommerce.monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter(config_market=config_market)
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example() >= -5000


full_episode_testcases_rl_agent = [
	(linear_market.LinearEconomyDuopoly(config=config_market), QLearningAgent, 'LinearEconomyDuopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyMonopoly(config=config_market), QLearningAgent,
		'CircularEconomyMonopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceMonopoly(config=config_market), QLearningAgent,
		'CircularEconomyRebuyPriceMonopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_market), QLearningAgent,
		'CircularEconomyRebuyPriceDuopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_market), ContinuousActorCriticAgentFixedOneStd,
		'actor_parametersCircularEconomyRebuyPriceDuopoly_ContinuousActorCriticAgentFixedOneStd.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_market), DiscreteActorCriticAgent,
		'actor_parametersCircularEconomyRebuyPriceDuopoly_DiscreteACACircularEconomyRebuy.dat')
]


@pytest.mark.parametrize('marketplace, agent_class, parameters_file', full_episode_testcases_rl_agent)
def test_full_episode_rl_agents(marketplace, agent_class, parameters_file):
	agent = agent_class(
		marketplace=marketplace,
		config_market=config_market,
		config_rl=config_rl,
		load_path=os.path.join(parameters_path, parameters_file))
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'),\
		patch('recommerce.monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter(config_market=config_market)
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example() >= -5000


@pytest.mark.slow
def test_exampleprinter_with_tensorboard():
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'):
		assert ExamplePrinter(config_market=config_market).run_example() >= -5000
