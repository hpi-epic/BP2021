import os
import re
import shutil

import pytest

import sim_market as sim
import vendors

from .context import exampleprinter


def teardown_module(module):
	print('***TEARDOWN***')
	for f in os.listdir('./runs'):
		if re.match('test_*', f):
			shutil.rmtree('./runs/' + f)


test_scenarios = [(sim.ClassicScenario(), vendors.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), vendors.FixedPriceLEAgent()), (sim.CircularEconomy(), vendors.FixedPriceCEAgent()), (sim.CircularEconomy(), vendors.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPrice(), vendors.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPrice(), vendors.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_full_episode(environment, agent):
	assert exampleprinter.print_example(environment, agent, log_dir_prepend='test_') >= -5000
