import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer

test_scenarios = [
	(linear_market.LinearEconomyDuopoly, actorcritic_agent.DiscreteActorCriticAgent, True),
	(linear_market.LinearEconomyDuopoly, actorcritic_agent.ContinuousActorCriticAgentFixedOneStd, True),
	(linear_market.LinearEconomyDuopoly, actorcritic_agent.ContinuousActorCriticAgentEstimatingStd, False),
	(linear_market.LinearEconomyOligopoly, actorcritic_agent.DiscreteActorCriticAgent, False),
	(linear_market.LinearEconomyOligopoly, actorcritic_agent.ContinuousActorCriticAgentFixedOneStd, False),
	(linear_market.LinearEconomyOligopoly, actorcritic_agent.ContinuousActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyMonopoly, actorcritic_agent.DiscreteActorCriticAgent, True),
	(circular_market.CircularEconomyMonopoly, actorcritic_agent.ContinuousActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyMonopoly, actorcritic_agent.ContinuousActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceMonopoly, actorcritic_agent.DiscreteActorCriticAgent, True),
	(circular_market.CircularEconomyRebuyPriceMonopoly, actorcritic_agent.ContinuousActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyRebuyPriceMonopoly, actorcritic_agent.ContinuousActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceDuopoly, actorcritic_agent.DiscreteActorCriticAgent, False),
	(circular_market.CircularEconomyRebuyPriceDuopoly, actorcritic_agent.ContinuousActorCriticAgentFixedOneStd, True),
	(circular_market.CircularEconomyRebuyPriceDuopoly, actorcritic_agent.ContinuousActorCriticAgentEstimatingStd, False)
]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('market_class, agent_class, verbose', test_scenarios)
def test_training_configurations(market_class, agent_class, verbose):
	ActorCriticTrainer(market_class, agent_class).train_agent(
		verbose=verbose,
		number_of_training_steps=120,
		total_envs=64)
