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


test_scenarios = [(sim.ClassicScenario(), agent.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), agent.FixedPriceLEAgent()), (sim.CircularEconomy(), agent.FixedPriceCEAgent()), (sim.CircularEconomy(), agent.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPrice(), agent.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPrice(), agent.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_full_episode(environment, agent):
	assert exampleprinter.print_example(environment, agent, log_dir_prepend='test_') >= -5000
