import os
import re
import shutil
import time
from unittest.mock import patch

import pytest

import agents.vendors as vendors
import market.linear_market.linear_sim_market as linear_sim_market
import market.circular_market.circular_sim_market as circular_sim_market
from monitoring.exampleprinter import ExamplePrinter


def test_setup_exampleprinter():
	printer = ExamplePrinter()
	printer.setup_exampleprinter(marketplace=linear_sim_market.ClassicScenario(), agent=vendors.FixedPriceLEAgent())
	assert isinstance(printer.marketplace, linear_sim_market.ClassicScenario)
	assert isinstance(printer.agent, vendors.FixedPriceLEAgent)


full_episode_testcases = [
	(linear_sim_market.ClassicScenario(), vendors.FixedPriceLEAgent()),
	(linear_sim_market.MultiCompetitorScenario(), vendors.FixedPriceLEAgent()),
	(circular_sim_market.CircularEconomyMonopolyScenario(), vendors.FixedPriceCEAgent()),
	(circular_sim_market.CircularEconomyMonopolyScenario(), vendors.RuleBasedCEAgent()),
	(circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario(), vendors.FixedPriceCERebuyAgent()),
	(circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario(), vendors.RuleBasedCERebuyAgent()),
	(circular_sim_market.CircularEconomyRebuyPriceOneCompetitor(), vendors.FixedPriceCERebuyAgent()),
	(circular_sim_market.CircularEconomyRebuyPriceOneCompetitor(), vendors.RuleBasedCERebuyAgent())
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
