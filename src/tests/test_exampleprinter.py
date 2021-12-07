import pytest

import agent
import sim_market as sim

from .context import exampleprinter

test_scenarios = [(sim.ClassicScenario(), agent.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), agent.FixedPriceLEAgent()), (sim.CircularEconomy(), agent.FixedPriceCEAgent()), (sim.CircularEconomy(), agent.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPrice(), agent.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPrice(), agent.RuleBasedCERebuyAgent())]
@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_full_episode(environment, agent):
    assert exampleprinter.print_example(environment, agent) >= -5000
