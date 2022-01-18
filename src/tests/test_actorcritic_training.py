import pytest

import market.sim_market as sim_market
import rl.actorcritic_agent as a2c_agent
import rl.actorcritic_training as a2c_training


def test_standard_setup():
    a2c_training.train_actorcritic()


test_scenarios = [
    (sim_market.ClassicScenario, a2c_agent.DiscreteACALinear),
    (sim_market.ClassicScenario, a2c_agent.ContinuosActorCriticAgent)
]


@pytest.mark.parametrize('marketplace, agent', test_scenarios)
def test_training_configurations(marketplace, agent):
    a2c_training.train_actorcritic(marketplace_class=marketplace, agent_class=agent)
