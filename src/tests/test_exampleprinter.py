import os
import re
import shutil
import time
from unittest.mock import patch

import pytest

import agents.vendors as vendors
import market.circular.circular_sim_market as circular_market
import market.linear.linear_sim_market as linear_market
import rl.actorcritic_agent as actorcritic_agent
from monitoring.exampleprinter import ExamplePrinter


def test_setup_exampleprinter():
	printer = ExamplePrinter()
	printer.setup_exampleprinter(marketplace=linear_market.ClassicScenario(), agent=vendors.FixedPriceLEAgent())
	assert isinstance(printer.marketplace, linear_market.ClassicScenario)
	assert isinstance(printer.agent, vendors.FixedPriceLEAgent)


parameters_path = os.path.join('results', 'monitoring')
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
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), vendors.FixedPriceCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), vendors.RuleBasedCERebuyAgent()),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor(), actorcritic_agent.ContinuosActorCriticAgentFixedOneStd(6, 3,
		os.path.join(parameters_path, 'actor_parametersCircularRebuyOneComp_ContinuosA2C.dat')))
]


@pytest.mark.parametrize('marketplace, agent', full_episode_testcases)
def test_full_episode(marketplace, agent):
	with patch('monitoring.exampleprinter.SVGManipulator'),\
		patch('monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter()
		printer.setup_exampleprinter(marketplace, agent)
		assert printer.run_example(log_dir_prepend='test_') >= -5000


def test_exampleprinter_with_tensorboard():
	with patch('monitoring.exampleprinter.SVGManipulator'):
		assert ExamplePrinter().run_example(log_dir_prepend='test_') >= -5000

	print('***TEARDOWN***')
	# we need to sleep because sometimes the runs folder is still being used when we try to remove it
	time.sleep(0.002)
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs')):
		if re.match('test_*', file_name):
			shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs', file_name))
