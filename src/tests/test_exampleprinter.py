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


test_cases = [(sim.ClassicScenario(), vendors.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), vendors.FixedPriceLEAgent()), (sim.CircularEconomyMonopolyScenario(), vendors.FixedPriceCEAgent()), (sim.CircularEconomyMonopolyScenario(), vendors.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), vendors.RuleBasedCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), vendors.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', test_cases)
def test_full_episode(environment, agent):
	assert exampleprinter.run_example(environment, agent, log_dir_prepend='test_') >= -5000
