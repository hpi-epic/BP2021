import os
import re
import shutil
import time
from unittest.mock import patch

import pytest

import agents.vendors as vendors
import market.sim_market as sim
from monitoring.exampleprinter import ExamplePrinter

test_cases = [
	(sim.ClassicScenario(), vendors.FixedPriceLEAgent()),
	(sim.MultiCompetitorScenario(), vendors.FixedPriceLEAgent()),
	(sim.CircularEconomyMonopolyScenario(), vendors.FixedPriceCEAgent()),
	(sim.CircularEconomyMonopolyScenario(), vendors.RuleBasedCEAgent()),
	(sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.FixedPriceCERebuyAgent()),
	(sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.RuleBasedCERebuyAgent()),
	(sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.FixedPriceCERebuyAgent()),
	(sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.RuleBasedCERebuyAgent())
]


@pytest.mark.parametrize('environment, agent', test_cases)
def test_full_episode(environment, agent):
	with patch('monitoring.exampleprinter.SVGManipulator'),\
		patch('monitoring.exampleprinter.SummaryWriter'):
		printer = ExamplePrinter()
		printer.setup_exampleprinter(environment, agent)
		assert printer.run_example(log_dir_prepend='test_') >= -5000


def test_exampleprinter_with_tensorboard():
	with patch('monitoring.exampleprinter.SVGManipulator'):
		assert ExamplePrinter().run_example(log_dir_prepend='test_') >= -5000

	print('***TEARDOWN***')
	# we need to sleep because sometimes the runs folder is still being used when we try to remove it
	time.sleep(0.001)
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs')):
		if re.match('test_*', file_name):
			shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'results', 'runs', file_name))
	# remove the runs folder if it is empty, because that means it has only been created for our tests
	if os.listdir(os.path.join('results', 'runs')) == []:
		os.rmdir(os.path.join('results', 'runs'))
