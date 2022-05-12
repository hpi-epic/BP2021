import pytest
import utils_tests as ut_t

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer

config_hyperparameter: HyperparameterConfig = ut_t.mock_config_hyperparameter()

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
	ActorCriticTrainer(market_class, agent_class, config=ut_t.mock_config_hyperparameter()).train_agent(
		verbose=verbose,
		number_of_training_steps=120,
		total_envs=64)
