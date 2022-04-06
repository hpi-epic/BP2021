from unittest.mock import patch

import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer
from recommerce.market.linear.linear_vendors import CompetitorJust2Players, CompetitorLinearRatio1, CompetitorRandom
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgentCompetitive

multi_competitors = [CompetitorLinearRatio1(), CompetitorRandom(), CompetitorJust2Players()]

test_scenarios = [
	(linear_market.LinearEconomy(competitors=CompetitorLinearRatio1()), actorcritic_agent.DiscreteACALinear, True),
	(linear_market.LinearEconomy(competitors=CompetitorLinearRatio1()), actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(linear_market.LinearEconomy(competitors=CompetitorLinearRatio1()), actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False),
	(linear_market.LinearEconomy(competitors=multi_competitors), actorcritic_agent.DiscreteACALinear, False),
	(linear_market.LinearEconomy(competitors=multi_competitors), actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(linear_market.LinearEconomy(competitors=multi_competitors), actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomy(competitors=[]), actorcritic_agent.DiscreteACACircularEconomy, True),
	(circular_market.CircularEconomy(competitors=[]), actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomy(competitors=[]), actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPrice(competitors=[]), actorcritic_agent.DiscreteACACircularEconomyRebuy, True),
	(circular_market.CircularEconomyRebuyPrice(competitors=[]), actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyRebuyPrice(competitors=[]), actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPrice(competitors=RuleBasedCERebuyAgentCompetitive()), actorcritic_agent.DiscreteACACircularEconomyRebuy, False),
	(circular_market.CircularEconomyRebuyPrice(competitors=RuleBasedCERebuyAgentCompetitive()), actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(circular_market.CircularEconomyRebuyPrice(competitors=RuleBasedCERebuyAgentCompetitive()), actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False)
]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('market, agent_class, verbose', test_scenarios)
def test_training_configurations(market, agent_class, verbose):
	with patch('recommerce.rl.training.SummaryWriter'):
		ActorCriticTrainer(market, agent_class, log_dir_prepend='test_').train_agent(
			verbose=verbose,
			number_of_training_steps=120,
			total_envs=64)
