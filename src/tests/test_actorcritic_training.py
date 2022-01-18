import pytest

import market.sim_market as sim_market
import rl.actorcritic_agent as a2c_agent
import rl.actorcritic_training as a2c_training


def test_standard_setup():
    a2c_training.train_actorcritic()


test_scenarios = [
    (sim_market.ClassicScenario, a2c_agent.DiscreteACALinear, 10),
    (sim_market.ClassicScenario, a2c_agent.ContinuosActorCriticAgent, 1)
]


@pytest.mark.parametrize('marketplace, agent, outputs', test_scenarios)
def test_training_configurations(marketplace, agent, outputs):
    a2c_training.train_actorcritic(marketplace_class=marketplace, agent_class=agent, outputs=outputs)
