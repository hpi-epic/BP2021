import os
from unittest.mock import patch

import pytest

import alpha_business.agents.vendors as vendors
import alpha_business.market.circular.circular_sim_market as circular_market
import alpha_business.market.linear.linear_sim_market as linear_market
import alpha_business.rl.actorcritic_agent as actorcritic_agent
from alpha_business.monitoring.exampleprinter import ExamplePrinter

# The load path for the agent modelfiles
parameters_path = os.path.join('tests', 'test_data')


def test_setup_exampleprinter():
	printer = ExamplePrinter()
	printer.setup_exampleprinter(marketplace=linear_market.ClassicScenario(), agent=vendors.FixedPriceLEAgent())
	assert isinstance(printer.marketplace, linear_market.ClassicScenario)
	assert isinstance(printer.agent, vendors.FixedPriceLEAgent)


full_episode_testcases = [
	(linear_market.ClassicScenario(), vendors.FixedPriceLEAgent()),
	(linear_market.ClassicScenario(), vendors.QLearningLEAgent(3, 10,
		load_path=os.path.join(parameters_path, 'ClassicScenario_QLearningLEAgent.dat'))),
	(linear_market.MultiCompetitorScenario(), vendors.FixedPriceLEAgent()),
	(circular_market.CircularEconomyMonopolyScenario(), vendors.FixedPriceCEAgent()),
	(circular_market.CircularEconomyMonopolyScenario(), vendors.RuleBasedCEAgent()),
	(circular_market.CircularEconomyMonopolyScenario(), vendors.QLearningCEAgent(2, 100,
		load_path=os.path.join(parameters_path, 'CircularEconomyMonopolyScenario_QLearningCEAgent.dat'))),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario(), vendors.FixedPriceCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario(), vendors.RuleBasedCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario(), vendors.QLearningCERebuyAgent(2, 1000,
		load_path=os.path.join(parameters_path, 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'))),
	# (circular_market.CircularEconomyRebuyPriceMonopolyScenario(), actorcritic_agent.ContinuosActorCriticAgentEstimatingStd(2, 3,
	# 	load_path=os.path.join(parameters_path,
	# 		'actor_parametersCircularEconomyRebuyPriceMonopolyScenario_ContinuosActorCriticAgentEstimatingStd.dat'))),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), vendors.FixedPriceCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), vendors.RuleBasedCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), vendors.QLearningCERebuyAgent(6, 1000,
		load_path=os.path.join(parameters_path, 'CircularEconomyRebuyPriceOneCompetitor_QLearningCERebuyAgent.dat'))),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), actorcritic_agent.ContinuosActorCriticAgentFixedOneStd(6, 3,
		load_path=os.path.join(parameters_path,
			'actor_parametersCircularEconomyRebuyPriceOneCompetitor_ContinuosActorCriticAgentFixedOneStd.dat'))),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), actorcritic_agent.DiscreteACACircularEconomyRebuy(6, 1000,
		load_path=os.path.join(parameters_path, 'actor_parametersCircularEconomyRebuyPriceOneCompetitor_DiscreteACACircularEconomyRebuy.dat')))
]


@pytest.mark.parametrize('marketplace, agent', full_episode_testcases)
def test_full_episode(marketplace, agent):
	with patch('alpha_business.monitoring.exampleprinter.SVGManipulator'),\
		patch('alpha_business.monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter()
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example(log_dir_prepend='test_') >= -5000


def test_exampleprinter_with_tensorboard():
	with patch('alpha_business.monitoring.exampleprinter.SVGManipulator'):
		assert ExamplePrinter().run_example(log_dir_prepend='test_') >= -5000