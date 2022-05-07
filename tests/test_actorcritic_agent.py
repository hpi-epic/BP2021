import pytest
import torch
import utils_tests as ut_t

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import HyperparameterConfig

config_hyperparameter: HyperparameterConfig = ut_t.mock_config_hyperparameter()

abstract_agent_classes_testcases = [
	actorcritic_agent.ActorCriticAgent,
	actorcritic_agent.ContinuosActorCriticAgent
]


@pytest.mark.parametrize('agent', abstract_agent_classes_testcases)
def test_abstract_agent_classes(agent):
	with pytest.raises(TypeError) as error_message:
		agent(linear_market.LinearEconomyDuopoly)
	assert 'Can\'t instantiate abstract class' in str(error_message.value)


marketplace_classes = [
	linear_market.LinearEconomyDuopoly,
	linear_market.LinearEconomyOligopoly,
	circular_market.CircularEconomyMonopoly,
	circular_market.CircularEconomyRebuyPriceMonopoly,
	circular_market.CircularEconomyRebuyPriceDuopoly
]


@pytest.mark.parametrize('market_class', marketplace_classes)
def test_discrete_agents_initializes_networks_correct(market_class):
	marketplace = market_class(config=config_hyperparameter)
	agent = actorcritic_agent.DiscreteActorCriticAgent(marketplace=marketplace, config=config_hyperparameter)
	assert agent.actor_net is not None
	assert agent.critic_net is not None
	assert agent.critic_tgt_net is not None
	input_size = marketplace.get_observations_dimension()
	output_size = marketplace.get_n_actions()
	test_input = torch.ones(input_size).to(agent.device)
	actor_output = agent.actor_net(test_input)
	assert len(actor_output.to('cpu')) == output_size


@pytest.mark.parametrize('market_class', marketplace_classes)
def test_continous_agents_initializes_networks_correct(market_class):
	marketplace = market_class(config=config_hyperparameter)
	agent = actorcritic_agent.ContinuosActorCriticAgentFixedOneStd(marketplace=marketplace, config=config_hyperparameter)
	assert agent.actor_net is not None
	assert agent.critic_net is not None
	assert agent.critic_tgt_net is not None
	input_size = marketplace.get_observations_dimension()
	output_size = marketplace.get_actions_dimension()
	test_input = torch.ones(input_size).to(agent.device)
	actor_output = agent.actor_net(test_input)
	assert len(actor_output.to('cpu')) == output_size


@pytest.mark.parametrize('market_class', marketplace_classes)
def test_std_estimating_agents_initializes_networks_correct(market_class):
	marketplace = market_class(config=config_hyperparameter)
	agent = actorcritic_agent.ContinuosActorCriticAgentEstimatingStd(marketplace=marketplace, config=config_hyperparameter)
	assert agent.actor_net is not None
	assert agent.critic_net is not None
	assert agent.critic_tgt_net is not None
	input_size = marketplace.get_observations_dimension()
	output_size = marketplace.get_actions_dimension()
	test_input = torch.ones(input_size).to(agent.device)
	actor_output = agent.actor_net(test_input)
	assert len(actor_output.to('cpu')) == 2 * output_size


agent_initialization_testcases = [
	actorcritic_agent.DiscreteActorCriticAgent,
	actorcritic_agent.ContinuosActorCriticAgentFixedOneStd,
	actorcritic_agent.ContinuosActorCriticAgentFixedOneStd
]


@pytest.mark.parametrize(
	'agent_class, market_class',
	ut.cartesian_product(agent_initialization_testcases, marketplace_classes)
)
def test_agents_generate_valid_actions(agent_class, market_class):
	marketplace = market_class(config=config_hyperparameter)
	agent = agent_class(marketplace=marketplace, config=config_hyperparameter)
	test_input = marketplace.observation_space.sample()
	action = agent.policy(test_input)
	next_state, reward, done, info = marketplace.step(action)
	assert isinstance(reward, float)
	assert isinstance(done, bool)
	assert isinstance(info, dict)
	critic_output = agent.critic_net(torch.from_numpy(next_state).to(agent.device))
	assert isinstance(critic_output.to('cpu').item(), float)
