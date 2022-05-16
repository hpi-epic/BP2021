import os
from unittest.mock import patch

import pytest
import utils_tests as ut_t
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.circular.circular_vendors as circular_vendors
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.market.linear.linear_vendors import FixedPriceLEAgent
from recommerce.monitoring.exampleprinter import ExamplePrinter
from recommerce.rl.actorcritic.actorcritic_agent import ContinuosActorCriticAgentFixedOneStd, DiscreteActorCriticAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

# The load path for the agent modelfiles
parameters_path = os.path.join('tests', 'test_data')

config_hyperparameter: AttrDict = ut_t.mock_config_hyperparameter()


def test_setup_exampleprinter():
	printer = ExamplePrinter(config=config_hyperparameter)
	printer.setup_exampleprinter(
		marketplace=linear_market.LinearEconomyDuopoly(config=config_hyperparameter),
		agent=FixedPriceLEAgent(config=config_hyperparameter)
		)
	assert isinstance(printer.marketplace, linear_market.LinearEconomyDuopoly)
	assert isinstance(printer.agent, FixedPriceLEAgent)


full_episode_testcases_rule_based = [
	(linear_market.LinearEconomyDuopoly(config=config_hyperparameter),
		FixedPriceLEAgent(config=config_hyperparameter)),
	(linear_market.LinearEconomyOligopoly(config=config_hyperparameter),
		FixedPriceLEAgent(config=config_hyperparameter)),
	(circular_market.CircularEconomyMonopoly(config=config_hyperparameter),
		circular_vendors.FixedPriceCEAgent(config=config_hyperparameter)),
	(circular_market.CircularEconomyMonopoly(config=config_hyperparameter),
		circular_vendors.RuleBasedCEAgent(config=config_hyperparameter)),
	(circular_market.CircularEconomyRebuyPriceMonopoly(config=config_hyperparameter),
		circular_vendors.FixedPriceCERebuyAgent(config=config_hyperparameter)),
	(circular_market.CircularEconomyRebuyPriceMonopoly(config=config_hyperparameter),
		circular_vendors.RuleBasedCERebuyAgent(config=config_hyperparameter)),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter),
		circular_vendors.FixedPriceCERebuyAgent(config=config_hyperparameter)),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter),
		circular_vendors.RuleBasedCERebuyAgent(config=config_hyperparameter))
]


@pytest.mark.parametrize('marketplace, agent', full_episode_testcases_rule_based)
def test_full_episode_rule_based(marketplace, agent):
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'),\
		patch('recommerce.monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter(config=config_hyperparameter)
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example() >= -5000


full_episode_testcases_rl_agent = [
	(linear_market.LinearEconomyDuopoly(config=config_hyperparameter), QLearningAgent, 'LinearEconomyDuopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyMonopoly(config=config_hyperparameter), QLearningAgent,
		'CircularEconomyMonopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceMonopoly(config=config_hyperparameter), QLearningAgent,
		'CircularEconomyRebuyPriceMonopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter), QLearningAgent,
		'CircularEconomyRebuyPriceDuopoly_QLearningAgent.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter), ContinuosActorCriticAgentFixedOneStd,
		'actor_parametersCircularEconomyRebuyPriceDuopoly_ContinuosActorCriticAgentFixedOneStd.dat'),
	(circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter), DiscreteActorCriticAgent,
		'actor_parametersCircularEconomyRebuyPriceDuopoly_DiscreteACACircularEconomyRebuy.dat')
]


@pytest.mark.parametrize('marketplace, agent_class, parameters_file', full_episode_testcases_rl_agent)
def test_full_episode_rl_agents(marketplace, agent_class, parameters_file):
	agent = agent_class(marketplace=marketplace, config=config_hyperparameter, load_path=os.path.join(parameters_path, parameters_file))
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'),\
		patch('recommerce.monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter(config=config_hyperparameter)
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example() >= -5000


@pytest.mark.slow
def test_exampleprinter_with_tensorboard():
	with patch('recommerce.monitoring.exampleprinter.SVGManipulator'):
		assert ExamplePrinter(config=config_hyperparameter).run_example() >= -5000
