from unittest.mock import patch

import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer

test_scenarios = [
	(linear_market.ClassicScenario, actorcritic_agent.DiscreteACALinear, True),
	(linear_market.ClassicScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(linear_market.ClassicScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False),
	(linear_market.MultiCompetitorScenario, actorcritic_agent.DiscreteACALinear, False),
	(linear_market.MultiCompetitorScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(linear_market.MultiCompetitorScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomy, True),
	(circular_market.CircularEconomyMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomyRebuy, True),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.DiscreteACACircularEconomyRebuy, False),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False)
]


@pytest.mark.parametrize('marketplace, agent, verbose', test_scenarios)
def test_training_configurations(marketplace, agent, verbose):
	with patch('recommerce.rl.training.SummaryWriter'), \
		patch('recommerce.rl.actorcritic.actorcritic_agent.ActorCriticAgent.save'):
		ActorCriticTrainer(marketplace, agent, log_dir_prepend='test_').train_agent(
			verbose=verbose,
			number_of_training_steps=120,
			total_envs=64)
