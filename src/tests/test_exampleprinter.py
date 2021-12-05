import pytest

from src.sim_market import (CircularEconomy, CircularEconomyRebuyPrice,
                            ClassicScenario, MultiCompetitorScenario)

from .context import (FixedPriceCEAgent, FixedPriceCERebuyAgent,
                      FixedPriceLEAgent, RuleBasedCEAgent,
                      RuleBasedCERebuyAgent, exampleprinter)

test_scenarios = [(ClassicScenario(), FixedPriceLEAgent()), (MultiCompetitorScenario(), FixedPriceLEAgent()), (CircularEconomy(), FixedPriceCEAgent()), (CircularEconomy(), RuleBasedCEAgent()), (CircularEconomyRebuyPrice(), FixedPriceCERebuyAgent()), (CircularEconomyRebuyPrice(), RuleBasedCERebuyAgent())]
@pytest.mark.parametrize('environment, agent', test_scenarios)
def test_full_episode(environment, agent):
    assert exampleprinter.print_example(environment, agent) >= -5000
