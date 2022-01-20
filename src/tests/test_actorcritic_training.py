from unittest.mock import patch

import pytest

import market.sim_market as sim_market
import rl.actorcritic_agent as actorcritic_agent
import rl.actorcritic_training as actorcritic_training


def test_standard_setup():
	with patch('rl.actorcritic_training.SummaryWriter'):
		actorcritic_training.train_actorcritic()


test_scenarios = [
	(sim_market.ClassicScenario, actorcritic_agent.DiscreteACALinear, True),
	(sim_market.ClassicScenario, actorcritic_agent.ContinuosActorCriticAgent, True),
	(sim_market.MultiCompetitorScenario, actorcritic_agent.DiscreteACALinear, False),
	(sim_market.MultiCompetitorScenario, actorcritic_agent.ContinuosActorCriticAgent, False),
	(sim_market.CircularEconomyMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomy, True),
	(sim_market.CircularEconomyMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgent, False),
	(sim_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomyRebuy, True),
	(sim_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgent, False),
	(sim_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.DiscreteACACircularEconomyRebuy, False),
	(sim_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgent, True)
]


@pytest.mark.parametrize('marketplace, agent, verbose', test_scenarios)
def test_training_configurations(marketplace, agent, verbose):
	with patch('rl.actorcritic_training.SummaryWriter'):
		actorcritic_training.train_actorcritic(marketplace_class=marketplace, agent_class=agent, verbose=verbose, number_of_training_steps=120, total_envs=64)
