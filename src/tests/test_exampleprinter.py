import os
import re
import shutil

import pytest

import agent
import exampleprinter
import sim_market as sim


def teardown_module(module):
	print('***TEARDOWN***')
	for f in os.listdir('./runs'):
		if re.match('test_*', f):
			shutil.rmtree('./runs/' + f)


test_cases = [(sim.ClassicScenario(), agent.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), agent.FixedPriceLEAgent()), (sim.CircularEconomyMonopolyScenario(), agent.FixedPriceCEAgent()), (sim.CircularEconomyMonopolyScenario(), agent.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), agent.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), agent.RuleBasedCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), agent.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceOneCompetitor(), agent.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', test_cases)
def test_full_episode(environment, agent):
	assert exampleprinter.run_example(environment, agent, log_dir_prepend='test_') >= -5000
