import os
import re
import shutil

import pytest

import agent
import sim_market as sim

from .context import exampleprinter


def teardown_module(module):
	print('***TEARDOWN***')
	for f in os.listdir('./runs'):
		if re.match('test_*', f):
			shutil.rmtree('./runs/' + f)


testcombinations = [(sim.ClassicScenario(), agent.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), agent.FixedPriceLEAgent()), (sim.CircularEconomyMonopolyScenario(), agent.FixedPriceCEAgent()), (sim.CircularEconomyMonopolyScenario(), agent.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), agent.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), agent.RuleBasedCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), agent.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), agent.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', testcombinations)
def test_full_episode(environment, agent):
	assert exampleprinter.print_example(environment, agent, log_dir_prepend='test_') >= -5000
