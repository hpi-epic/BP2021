import pytest

import market.sim_market as sim_market
import rl.actorcritic_agent as a2c_agent
import rl.actorcritic_training as a2c_training


def test_standard_setup():
    a2c_training.train_actorcritic()


test_scenarios = [
    (sim_market.ClassicScenario, a2c_agent.DiscreteACALinear, True),
    (sim_market.ClassicScenario, a2c_agent.ContinuosActorCriticAgent, True),
    (sim_market.MultiCompetitorScenario, a2c_agent.DiscreteACALinear, False),
    (sim_market.MultiCompetitorScenario, a2c_agent.ContinuosActorCriticAgent, False),
    (sim_market.CircularEconomyMonopolyScenario, a2c_agent.DiscreteACACircularEconomy, True),
    (sim_market.CircularEconomyMonopolyScenario, a2c_agent.ContinuosActorCriticAgent, False),
    (sim_market.CircularEconomyRebuyPriceMonopolyScenario, a2c_agent.DiscreteACACircularEconomyRebuy, True),
    (sim_market.CircularEconomyRebuyPriceMonopolyScenario, a2c_agent.ContinuosActorCriticAgent, False),
    (sim_market.CircularEconomyRebuyPriceOneCompetitor, a2c_agent.DiscreteACACircularEconomyRebuy, False),
    (sim_market.CircularEconomyRebuyPriceOneCompetitor, a2c_agent.ContinuosActorCriticAgent, True)
]


@pytest.mark.parametrize('marketplace, agent, verbose', test_scenarios)
def test_training_configurations(marketplace, agent, verbose):
    a2c_training.train_actorcritic(marketplace_class=marketplace, agent_class=agent, verbose=verbose, number_of_training_steps=120, total_envs=64)
