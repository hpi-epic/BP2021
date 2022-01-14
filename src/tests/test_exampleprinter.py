import os
import re
import shutil
from unittest.mock import patch

import pytest

import agents.vendors as vendors
import market.sim_market as sim
import monitoring.exampleprinter as exampleprinter

test_cases = [(sim.ClassicScenario(), vendors.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), vendors.FixedPriceLEAgent()), (sim.CircularEconomyMonopolyScenario(), vendors.FixedPriceCEAgent()), (sim.CircularEconomyMonopolyScenario(), vendors.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.RuleBasedCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', test_cases)
def test_full_episode(environment, agent):
	with patch('monitoring.exampleprinter.SVGManipulator'),\
		patch('monitoring.exampleprinter.SummaryWriter'):
		assert exampleprinter.run_example(environment, agent, log_dir_prepend='test_') >= -5000


def test_exampleprinter_with_tensorboard():
	with patch('monitoring.exampleprinter.SVGManipulator'):
		assert exampleprinter.run_example(log_dir_prepend='test_') >= -5000

	for f in os.listdir('./runs'):
		if re.match('test_*', f):
			shutil.rmtree('./runs/' + f)
	if os.listdir('./runs') == []:
		os.rmdir('./runs')
