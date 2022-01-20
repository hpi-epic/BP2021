from abc import ABC, abstractmethod

import numpy as np
import torch

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut
import rl.model as model


class ActorCriticAgent(vendors.Agent, ABC):
	"""
	This is an implementation of an (one step) actor critic agent as proposed in Richard Suttons textbook on page 332.
	"""
	def __init__(self, n_observations, n_actions):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		print(f'I initiate an ActorCriticAgent using {self.device} device')
		self.initialize_models_and_optimizer(n_observations, n_actions)

	def synchronize_critic_tgt_net(self):
		"""
		This method writes the parameter from the critic net to it's target net.
		Call this method regularly during training.
		Having a target net solves problems occuring due to oscillation.
		"""
		print('Now I synchronize the tgt net')
		self.critic_tgt_net.load_state_dict(self.critic_net.state_dict())

	@abstractmethod
	def initialize_models_and_optimizer(self, n_observations, n_actions) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def policy(self, observation, verbose=False) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')

	def train_batch(self, states, actions, rewards, states_dash, regularization=False):
		states = states.to(self.device)
		actions = actions.to(self.device)
		rewards = rewards.to(self.device)
		states_dash = states_dash.to(self.device)
		self.critic_optimizer.zero_grad()
		self.actor_optimizer.zero_grad()

		v_estimates = self.critic_net(states)
		with torch.no_grad():
			v_expected = (rewards + config.GAMMA * self.critic_tgt_net(states_dash).detach()).view(-1, 1)
		valueloss = torch.nn.MSELoss()(v_estimates, v_expected)
		valueloss.backward()

		with torch.no_grad():
			baseline = v_estimates
			constant = (v_expected - baseline).detach()
		log_prob = -self.log_probability_given_action(states.detach(), actions.detach())
		policy_loss = torch.mean(constant * log_prob)
		if regularization:
			policy_loss += self.regularize(states)
		policy_loss.backward()

		self.critic_optimizer.step()
		self.actor_optimizer.step()

		return valueloss.to('cpu').item(), policy_loss.to('cpu').item()

	def regularize(self, states):
		"""
		Via regulation you can add punishment for unintended behaviour besides the reward.
		Use it to give "hints" to the agent or to improve stability.
		But be careful while using it.
		It could make the agent fulfill the regulation goals instead of maximizing the reward.

		Args:
			states (torch.Tensor): A tensor of the states the agent is in range

		Returns:
			torch.Tensor: The punishment for the agent
		"""
		return torch.zeros(1).squeeze().to(self.device)

	@abstractmethod
	def log_probability_given_action(self, states, actions) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def agent_output_to_market_form(self) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')


class DiscreteActorCriticAgent(ActorCriticAgent):
	"""
	This is an actor critic agent with discrete action space.
	It generates preferences and uses softmax to gain the probabilities.
	For our three markets we have three kinds of specific agents you must use.
	"""
	def initialize_models_and_optimizer(self, n_observations, n_actions):
		self.actor_net = model.simple_network(n_observations, n_actions).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0000025)
		self.critic_net = model.simple_network(n_observations, 1).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=0.00025)
		self.critic_tgt_net = model.simple_network(n_observations, 1).to(self.device)

	def policy(self, observation, verbose=False):
		observation = torch.Tensor(np.array(observation)).to(self.device)
		with torch.no_grad():
			distribution = torch.softmax(self.actor_net(observation).view(-1), dim=0)
			if verbose:
				v_estimat = self.critic_net(observation).view(-1)

		distribution = distribution.to('cpu').detach().numpy()
		action = ut.shuffle_from_probabilities(distribution)
		return action, distribution[action], v_estimat.to('cpu').item() if verbose else None

	def log_probability_given_action(self, states, actions):
		return -torch.log(torch.softmax(self.actor_net(states), dim=0).gather(1, actions.unsqueeze(-1)))


class DiscreteACALinear(DiscreteActorCriticAgent):
	def agent_output_to_market_form(self, action):
		return action


class DiscreteACACircularEconomy(DiscreteActorCriticAgent):
	def agent_output_to_market_form(self, action):
		return (int(action % config.MAX_PRICE), int(action / config.MAX_PRICE))


class DiscreteACACircularEconomyRebuy(DiscreteActorCriticAgent):
	def agent_output_to_market_form(self, action):
		return (int(action / (config.MAX_PRICE * config.MAX_PRICE)), int(action / config.MAX_PRICE % config.MAX_PRICE), int(action % config.MAX_PRICE))


class ContinuosActorCriticAgent(ActorCriticAgent):
	"""
	This is an actor critic agent with continuos action space.
	It's parametrization is a normal distribution with fixed mean.
	It works on any sort of market we have so far, just the number of action values must be given.
	"""
	softplus = torch.nn.Softplus()

	def initialize_models_and_optimizer(self, n_observations, n_actions):
		self.n_actions = n_actions
		self.actor_net = model.simple_network(n_observations, 2 * self.n_actions).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.00002)
		self.critic_net = model.simple_network(n_observations, 1).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=0.002)
		self.critic_tgt_net = model.simple_network(n_observations, 1).to(self.device)

	def policy(self, observation, verbose=False):
		observation = torch.Tensor(np.array(observation)).to(self.device)
		with torch.no_grad():
			network_result = self.softplus(self.actor_net(observation))
			network_result = network_result.view(2, -1)
			mean = network_result[0, :]
			std = network_result[1, :]
			if verbose:
				v_estimat = self.critic_net(observation).view(-1)

		action = torch.round(torch.normal(mean, std).to(self.device))
		action = torch.max(action, torch.zeros(action.shape).to(self.device))
		action = torch.min(action, 9 * torch.ones(action.shape).to(self.device))
		return action.squeeze().type(torch.LongTensor).to('cpu').numpy(), *((self.log_probability_given_action(observation, action).mean().detach().item(), v_estimat.to('cpu').item()) if verbose else (None, None))

	def log_probability_given_action(self, states, actions):
		network_result = self.softplus(self.actor_net(states))
		network_result = network_result.view(network_result.shape[0], 2, -1)
		mean = network_result[:, 0, :]
		std = network_result[:, 1, :]
		return torch.distributions.Normal(mean, std).log_prob(actions.view(-1, self.n_actions)).sum(dim=1).unsqueeze(-1)

	def regularize(self, states):
		"""
		This regularization pushes the actor with very high priority towards a mean price of 3.5.
		Use it at the beginning to avoid 0 pricing which gets only horrible negative reward.
		The magic number is a suitable constant to enforce quick movement to 3.5.

		Args:
			states (torch.Tensor): The current states the agent is in at the moment

		Returns:
			torch.Tensor: the malus of the regularization
		"""
		proposed_actions = self.actor_net(states.detach())
		return 10000 * torch.nn.MSELoss()(proposed_actions, 3.5 * torch.ones(proposed_actions.shape).to(self.device))

	def agent_output_to_market_form(self, action):
		return action.tolist()
