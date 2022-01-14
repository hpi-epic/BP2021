import os
import re
import shutil
from unittest.mock import patch

import pytest

import agents.vendors as vendors
import market.sim_market as sim
import monitoring.exampleprinter as exampleprinter


def teardown_module(module):
	print('***TEARDOWN***')
	for f in os.listdir('./runs'):
		if re.match('test_*', f):
			shutil.rmtree('./runs/' + f)


test_cases = [(sim.ClassicScenario(), vendors.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), vendors.FixedPriceLEAgent()), (sim.CircularEconomyMonopolyScenario(), vendors.FixedPriceCEAgent()), (sim.CircularEconomyMonopolyScenario(), vendors.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.RuleBasedCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', test_cases)
def test_full_episode(environment, agent):
	with patch('monitoring.exampleprinter.SVGManipulator'):
		# patch('monitoring.exampleprinter.SummaryWriter'):
		assert exampleprinter.run_example(environment, agent, log_dir_prepend='test_') >= -5000
