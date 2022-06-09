import pytest
import torch
from attrdict import AttrDict

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.actorcritic.actorcritic_agent import ContinuousActorCriticAgentFixedOneStd

config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceMonopoly)
config_rl: AttrDict = HyperparameterConfigLoader.load('actor_critic_config', ContinuousActorCriticAgentFixedOneStd)

abstract_agent_classes_testcases = [
	actorcritic_agent.ActorCriticAgent,
	actorcritic_agent.ContinuousActorCriticAgent
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
	marketplace = market_class(config=config_market)
	agent = actorcritic_agent.DiscreteActorCriticAgent(marketplace=marketplace, config_market=config_market, config_rl=config_rl)
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
	marketplace = market_class(config=config_market)
	agent = actorcritic_agent.ContinuousActorCriticAgentFixedOneStd(marketplace=marketplace, config_market=config_market, config_rl=config_rl)
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
	marketplace = market_class(config=config_market)
	agent = actorcritic_agent.ContinuousActorCriticAgentEstimatingStd(
		marketplace=marketplace, config_market=config_market, config_rl=config_rl)
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
	actorcritic_agent.ContinuousActorCriticAgentFixedOneStd,
	actorcritic_agent.ContinuousActorCriticAgentFixedOneStd
]


@pytest.mark.parametrize(
	'agent_class, market_class',
	ut.cartesian_product(agent_initialization_testcases, marketplace_classes)
)
def test_agents_generate_valid_actions(agent_class, market_class):
	marketplace = market_class(config=config_market)
	agent = agent_class(marketplace=marketplace, config_market=config_market, config_rl=config_rl)
	test_input = marketplace.observation_space.sample()
	action = agent.policy(test_input)
	next_state, reward, done, info = marketplace.step(action)
	assert isinstance(reward, float)
	assert isinstance(done, bool)
	assert isinstance(info, dict)
	critic_output = agent.critic_net(torch.from_numpy(next_state).to(agent.device))
	assert isinstance(critic_output.to('cpu').item(), float)
