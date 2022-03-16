import os
from unittest.mock import patch

import market.circular.circular_sim_market as circular_market
import market.linear.linear_sim_market as linear_market
import pytest
from market.circular.circular_vendors import FixedPriceCEAgent, FixedPriceCERebuyAgent, RuleBasedCEAgent, RuleBasedCERebuyAgent
from market.linear.linear_vendors import FixedPriceLEAgent
from monitoring.exampleprinter import ExamplePrinter
from rl.actorcritic.actorcritic_agent import ContinuosActorCriticAgentFixedOneStd, DiscreteACACircularEconomyRebuy
from rl.q_learning.q_learning_agent import QLearningCEAgent, QLearningCERebuyAgent, QLearningLEAgent

# The load path for the agent modelfiles
parameters_path = os.path.join('tests', 'test_data')


def teardown_module(module):
	print('***TEARDOWN***')
	# We need to sleep because sometimes the runs folder is still being used when we try to remove it.
	time.sleep(0.002)
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs')):
		if re.match('test_*', file_name):
			shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs', file_name))


def test_setup_exampleprinter():
	printer = ExamplePrinter()
	printer.setup_exampleprinter(marketplace=linear_market.ClassicScenario(), agent=FixedPriceLEAgent())
	assert isinstance(printer.marketplace, linear_market.ClassicScenario)
	assert isinstance(printer.agent, FixedPriceLEAgent)


full_episode_testcases = [
	(linear_market.ClassicScenario(), FixedPriceLEAgent()),
	(linear_market.ClassicScenario(), QLearningLEAgent(3, 10,
		load_path=os.path.join(parameters_path, 'ClassicScenario_QLearningLEAgent.dat'))),
	(linear_market.MultiCompetitorScenario(), FixedPriceLEAgent()),
	(circular_market.CircularEconomyMonopolyScenario(), FixedPriceCEAgent()),
	(circular_market.CircularEconomyMonopolyScenario(), RuleBasedCEAgent()),
	(circular_market.CircularEconomyMonopolyScenario(), QLearningCEAgent(2, 100,
		load_path=os.path.join(parameters_path, 'CircularEconomyMonopolyScenario_QLearningCEAgent.dat'))),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario(), FixedPriceCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario(), RuleBasedCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario(), QLearningCERebuyAgent(2, 1000,
		load_path=os.path.join(parameters_path, 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'))),
	# (circular_market.CircularEconomyRebuyPriceMonopolyScenario(), actorcritic_agent.ContinuosActorCriticAgentEstimatingStd(2, 3,
	# 	load_path=os.path.join(parameters_path,
	# 		'actor_parametersCircularEconomyRebuyPriceMonopolyScenario_ContinuosActorCriticAgentEstimatingStd.dat'))),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), FixedPriceCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), RuleBasedCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), QLearningCERebuyAgent(6, 1000,
		load_path=os.path.join(parameters_path, 'CircularEconomyRebuyPriceOneCompetitor_QLearningCERebuyAgent.dat'))),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), ContinuosActorCriticAgentFixedOneStd(6, 3,
		load_path=os.path.join(parameters_path,
			'actor_parametersCircularEconomyRebuyPriceOneCompetitor_ContinuosActorCriticAgentFixedOneStd.dat'))),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), DiscreteACACircularEconomyRebuy(6, 1000,
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
