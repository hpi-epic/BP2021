import pytest
import torch

import configuration.utils as ut
import rl.actorcritic_agent as actorcritic_agent

abstract_agent_classes_testcases = [
	actorcritic_agent.ActorCriticAgent,
	actorcritic_agent.DiscreteActorCriticAgent,
	actorcritic_agent.ContinuosActorCriticAgent
]


@pytest.mark.parametrize('actorcritic_agent', abstract_agent_classes_testcases)
def test_abstract_agent_classes(actorcritic_agent):
	with pytest.raises(TypeError):
		actorcritic_agent()


agent_initialization_testcases = [
	actorcritic_agent.DiscreteACALinear,
	actorcritic_agent.DiscreteACACircularEconomy,
	actorcritic_agent.DiscreteACACircularEconomyRebuy,
	actorcritic_agent.ContinuosActorCriticAgentFixedOneStd
]
input_sizes = [1, 3, 7, 19]
output_sizes_greater_zero = [2, 7, 15, 100, 1234]


@pytest.mark.parametrize('agent_class, input_output', ut.cartesian_product(agent_initialization_testcases, ut.cartesian_product(input_sizes, output_sizes_greater_zero)))
def test_agents_initializes_networks_correct_output_greater_zero(agent_class, input_output):
	input_size, output_size = input_output
	agent = agent_class(input_size, output_size)
	assert agent.actor_net is not None
	assert agent.critic_net is not None
	assert agent.critic_tgt_net is not None
	test_input = torch.ones(input_size).to(agent.device)
	actor_output = agent.actor_net(test_input)
	assert len(actor_output.to('cpu')) == output_size
	critic_output = agent.critic_net(test_input)
	assert isinstance(critic_output.to('cpu').item(), float)


@pytest.mark.parametrize('agent_class, input_size', ut.cartesian_product(agent_initialization_testcases, input_sizes))
def test_agents_initializes_network_correct_output_one(agent_class, input_size):
	agent = agent_class(input_size, 1)
	assert agent.actor_net is not None
	assert agent.critic_net is not None
	assert agent.critic_tgt_net is not None
	test_input = torch.ones(input_size).to(agent.device)
	actor_output = agent.actor_net(test_input)
	assert isinstance(actor_output.to('cpu').item(), float)
	critic_output = agent.critic_net(test_input)
	assert isinstance(critic_output.to('cpu').item(), float)


@pytest.mark.parametrize('input_size, output_size', ut.cartesian_product(input_sizes, output_sizes_greater_zero))
def test_std_estimating_agents_initializes_networks_correct_output_greater_zero(input_size, output_size):
	agent_class = actorcritic_agent.ContinuosActorCriticAgentEstimatingStd
	agent = agent_class(input_size, output_size)
	assert agent.actor_net is not None
	assert agent.critic_net is not None
	assert agent.critic_tgt_net is not None
	test_input = torch.ones(input_size).to(agent.device)
	actor_output = agent.actor_net(test_input)
	# For each parameter we need two outputs: One for the mean and one for the standard deviation
	assert len(actor_output.to('cpu')) == 2 * output_size
	critic_output = agent.critic_net(test_input)
	assert isinstance(critic_output.to('cpu').item(), float)


@pytest.mark.parametrize('input_size', input_sizes)
def test_std_estimating_agents_initializes_network_correct_output_one(input_size):
	agent_class = actorcritic_agent.ContinuosActorCriticAgentEstimatingStd
	agent = agent_class(input_size, 1)
	assert agent.actor_net is not None
	assert agent.critic_net is not None
	assert agent.critic_tgt_net is not None
	test_input = torch.ones(input_size).to(agent.device)
	actor_output = agent.actor_net(test_input)
	# We need exactly two outputs: One for the mean and one for the standard deviation
	assert len(actor_output.to('cpu')) == 2
	critic_output = agent.critic_net(test_input)
	assert isinstance(critic_output.to('cpu').item(), float)
