from unittest.mock import patch

import pytest

import market.circular_market.circular_sim_market as circular_sim_market
import market.linear_market.linear_sim_market as linear_sim_market
import rl.actorcritic_agent as actorcritic_agent
import rl.actorcritic_training as actorcritic_training


def test_standard_setup():
	with patch('rl.actorcritic_training.SummaryWriter'):
		actorcritic_training.train_actorcritic()


test_scenarios = [
	(linear_sim_market.ClassicScenario, actorcritic_agent.DiscreteACALinear, True),
	(linear_sim_market.ClassicScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(linear_sim_market.ClassicScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False),
	(linear_sim_market.MultiCompetitorScenario, actorcritic_agent.DiscreteACALinear, False),
	(linear_sim_market.MultiCompetitorScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(linear_sim_market.MultiCompetitorScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_sim_market.CircularEconomyMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomy, True),
	(circular_sim_market.CircularEconomyMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_sim_market.CircularEconomyMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomyRebuy, True),
	(circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_sim_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.DiscreteACACircularEconomyRebuy, False),
	(circular_sim_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(circular_sim_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False)
]


@pytest.mark.parametrize('marketplace, agent, verbose', test_scenarios)
def test_training_configurations(marketplace, agent, verbose):
	with patch('rl.actorcritic_training.SummaryWriter'):
		actorcritic_training.train_actorcritic(
			marketplace_class=marketplace,
			agent_class=agent,
			verbose=verbose,
			number_of_training_steps=120,
			total_envs=64)
