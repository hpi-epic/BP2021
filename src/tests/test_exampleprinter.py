import pytest

import agent
import sim_market as sim

from .context import exampleprinter

test_scenarios = [(sim.ClassicScenario(), agent.FixedPriceLEAgent()), (sim.MultiCompetitorScenario(), agent.FixedPriceLEAgent()), (sim.CircularEconomyMonopolyScenario(), agent.FixedPriceCEAgent()), (sim.CircularEconomyMonopolyScenario(), agent.RuleBasedCEAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), agent.FixedPriceCERebuyAgent()), (sim.CircularEconomyRebuyPriceMonopolyScenario(), agent.RuleBasedCERebuyAgent())]


@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_full_episode(environment, agent):
    assert exampleprinter.print_example(environment, agent) >= -5000
