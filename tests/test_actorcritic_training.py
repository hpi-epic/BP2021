import pytest
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer

test_scenarios = [
	(linear_market.LinearEconomyDuopoly, actorcritic_agent.DiscreteActorCriticAgent, True),
	(linear_market.LinearEconomyDuopoly, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(linear_market.LinearEconomyDuopoly, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False),
	(linear_market.LinearEconomyOligopoly, actorcritic_agent.DiscreteActorCriticAgent, False),
	(linear_market.LinearEconomyOligopoly, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(linear_market.LinearEconomyOligopoly, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyMonopoly, actorcritic_agent.DiscreteActorCriticAgent, True),
	(circular_market.CircularEconomyMonopoly, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyMonopoly, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceMonopoly, actorcritic_agent.DiscreteActorCriticAgent, True),
	(circular_market.CircularEconomyRebuyPriceMonopoly, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyRebuyPriceMonopoly, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceDuopoly, actorcritic_agent.DiscreteActorCriticAgent, False),
	(circular_market.CircularEconomyRebuyPriceDuopoly, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(circular_market.CircularEconomyRebuyPriceDuopoly, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False)
]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('market_class, agent_class, verbose', test_scenarios)
def test_training_configurations(market_class, agent_class, verbose):
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
	config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config')
	config_rl.batch_size = 8
	ActorCriticTrainer(market_class, agent_class, config_market, config_rl).train_agent(
		verbose=verbose,
		number_of_training_steps=120,
		total_envs=64)
