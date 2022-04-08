import os
from abc import ABC, abstractmethod

import numpy as np
import torch

import recommerce.configuration.utils as ut
import recommerce.rl.model as model
from recommerce.configuration.hyperparameter_config import config
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.market.sim_market import SimMarket
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class ActorCriticAgent(ReinforcementLearningAgent, ABC):
	"""
	This is an implementation of an (one step) actor critic agent as proposed in Richard Suttons textbook on page 332.
	"""
	def __init__(
			self,
			n_observations=None,
			n_actions=None,
			marketplace=None,
			optim=None,
			device='cuda' if torch.cuda.is_available() else 'cpu',
			load_path=None,
			critic_path=None,
			name='actor_critic',
			network_architecture=model.simple_network):
		assert marketplace is None or isinstance(marketplace, SimMarket), \
			f'if marketplace is provided, marketplace must be a SimMarket, but is {type(marketplace)}'
		assert (n_actions is None) == (n_observations is None), 'n_actions must be None exactly when n_observations is None'
		assert (n_actions is None) != (marketplace is None), \
			'You must specify the network size either by providing input and output size, or by a marketplace'

		if marketplace is not None:
			n_observations = marketplace.observation_space.shape[0]
			n_actions = marketplace.get_actions_dimension() if isinstance(self, ContinuosActorCriticAgent) else marketplace.get_n_actions()

		self.device = device
		self.name = name
		print(f'I initiate an ActorCriticAgent using {self.device} device')
		self.initialize_models_and_optimizer(n_observations, n_actions, network_architecture)
		if load_path is not None:
			self.actor_net.load_state_dict(torch.load(load_path, map_location=self.device))
		if critic_path is not None:
			self.critic_net.load_state_dict(torch.load(critic_path, map_location=self.device))
			self.critic_tgt_net.load_state_dict(torch.load(critic_path, map_location=self.device))

	def synchronize_tgt_net(self):
		# Not printing this anymore since it clutters the output when training
		# print('Now I synchronize the tgt net')
		self.critic_tgt_net.load_state_dict(self.critic_net.state_dict())

	@abstractmethod
	def initialize_models_and_optimizer(self, n_observations, n_actions) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def policy(self, observation, verbose=False, raw_action=False) -> None:  # pragma: no cover
		"""
		Give the current state to the agent and receive his action.

		Args:
			observation (torch.Tensor): The current observation
			verbose (bool, optional): Flag to add additional information about the training on the tensorboard.
			Defaults to False.
			raw_action (bool, optional): Flag to make the agent return his action without calling agent_output_to_market_form.
			Defaults to False.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

	def sync_to_best_interim(self):
		self.best_interim_actor_net.load_state_dict(self.actor_net.state_dict())
		self.best_interim_critic_net.load_state_dict(self.critic_net.state_dict())

	def save(self, model_path, model_name) -> None:
		"""
		Save a trained model to the specified folder within 'trainedModels'.
		For each model an actor and a critic net will be saved.
		This method is copied from our Q-Learning Agent

		Args:
			model_path (str): The path to the folder within 'trainedModels' where the model should be saved.
			model_name (str): The name of the .dat file of this specific model.
		"""
		assert model_name.endswith('.dat'), f'the modelname must end in ".dat": {model_name}'
		assert os.path.exists(model_path), f'the specified path does not exist: {model_path}'
		actor_path = os.path.join(model_path, f'actor_parameters{model_name}')
		torch.save(self.best_interim_actor_net.state_dict(), actor_path)
		torch.save(self.best_interim_critic_net.state_dict(), os.path.join(model_path, 'critic_parameters' + model_name))
		return actor_path

	def train_batch(self, states, actions, rewards, next_states, regularization=False):
		"""
		This is the main method to train both actor and critic network by a new batch.

		Args:
			states (torch.Tensor): Your current states
			actions (torch.Tensor): The actions you have taken
			rewards (torch.Tensor): The rewards you received from the marketplace
			next_states (torch.Tensor): The states you got into after your actions have been performed
			regularization (bool, optional): Do you want to use the regularization method? Defaults to False.

		Returns:
			float, float: the loss of your actor network and your critic network during this step
		"""
		states = states.to(self.device)
		actions = actions.to(self.device)
		rewards = rewards.to(self.device)
		next_states = next_states.to(self.device)
		self.critic_optimizer.zero_grad()
		self.actor_optimizer.zero_grad()

		v_estimates = self.critic_net(states)
		with torch.no_grad():
			v_expected = (rewards + config.gamma * self.critic_tgt_net(next_states).detach()).view(-1, 1)
		critic_loss = torch.nn.MSELoss()(v_estimates, v_expected)
		critic_loss.backward()

		with torch.no_grad():
			baseline = v_estimates
			constant = (v_expected - baseline).detach()
		log_prob = -self.log_probability_given_action(states.detach(), actions.detach())
		actor_loss = torch.mean(constant * log_prob)
		if regularization:
			actor_loss += self.regularize(states)
		actor_loss.backward()

		self.critic_optimizer.step()
		self.actor_optimizer.step()

		return actor_loss.to('cpu').item(), critic_loss.to('cpu').item()

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
	def agent_output_to_market_form(self, action) -> None:  # pragma: no cover
		"""
		Takes a raw action and transforms it to a form that is accepted by the market.
		A raw action is for example three numbers in one.

		Args:
			action (np.array or int): the raw action

		Returns:
			tuple or int: the action accepted by the market.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')


class DiscreteActorCriticAgent(ActorCriticAgent):
	"""
	This is an actor critic agent with discrete action space.
	It generates preferences and uses softmax to gain the probabilities.
	For our three markets we have three kinds of specific agents you must use.
	"""
	def initialize_models_and_optimizer(self, n_observations, n_actions, network_architecture):
		self.actor_net = network_architecture(n_observations, n_actions).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0000025)
		self.best_interim_actor_net = network_architecture(n_observations, n_actions).to(self.device)
		self.critic_net = network_architecture(n_observations, 1).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=0.00025)
		self.critic_tgt_net = network_architecture(n_observations, 1).to(self.device)
		self.best_interim_critic_net = self.critic_tgt_net = network_architecture(n_observations, 1).to(self.device)

	def policy(self, observation, verbose=False, raw_action=False):
		observation = torch.Tensor(np.array(observation)).to(self.device)
		with torch.no_grad():
			distribution = torch.softmax(self.actor_net(observation).view(-1), dim=0)
			if verbose:
				v_estimate = self.critic_net(observation).view(-1)

		distribution = distribution.to('cpu').detach().numpy()
		action = ut.shuffle_from_probabilities(distribution)
		action = action if raw_action else self.agent_output_to_market_form(action)

		if verbose:
			return action, distribution[action], v_estimate.to('cpu').item()
		else:
			return action

	def log_probability_given_action(self, states, actions):
		return -torch.log(torch.softmax(self.actor_net(states), dim=0).gather(1, actions.unsqueeze(-1)))


class DiscreteACALinear(DiscreteActorCriticAgent, LinearAgent):
	def agent_output_to_market_form(self, action):
		return action


class DiscreteACACircularEconomy(DiscreteActorCriticAgent, CircularAgent):
	def agent_output_to_market_form(self, action):
		return (int(action % config.max_price), int(action / config.max_price))


class DiscreteACACircularEconomyRebuy(DiscreteActorCriticAgent, CircularAgent):
	def agent_output_to_market_form(self, action):
		return (
			int(action / (config.max_price * config.max_price)),
			int(action / config.max_price % config.max_price),
			int(action % config.max_price))


class ContinuosActorCriticAgent(ActorCriticAgent, LinearAgent, CircularAgent):
	"""
	This is an actor critic agent with continuos action space.
	It's distribution is a normal distribution parameterized by mean and standard deviation.
	It works on any sort of market we have so far, just the number of action values must be given.
	Note that this class is abstract.
	You must use one of its subclasses.
	"""
	softplus = torch.nn.Softplus()
	name = 'ContinuosActorCriticAgent'

	def initialize_models_and_optimizer(self, n_observations, n_actions, network_architecture):
		self.n_actions = n_actions
		self.actor_net = network_architecture(n_observations, self.n_actions).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0002)
		self.best_interim_actor_net = network_architecture(n_observations, self.n_actions).to(self.device)
		self.critic_net = network_architecture(n_observations, 1).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=0.002)
		self.critic_tgt_net = network_architecture(n_observations, 1).to(self.device)
		self.best_interim_critic_net = network_architecture(n_observations, 1).to(self.device)

	@abstractmethod
	def transform_network_output(self, number_outputs, network_result):
		"""
		This method transforms the raw network output into an agent specific parametrization for mean and standard deviation.

		Args:
			number_outputs (int): the number of independent network outputs
			network_result (torch.Tensor): The output of your network. It will be used to generate your means and standard deviations.

		Returns:
			torch.Tensor, torch.Tensor: A tensor of your means and a tensor of your standard deviations
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

	def policy(self, observation, verbose=False, raw_action=False, mean_only=False):
		observation = torch.Tensor(np.array(observation)).to(self.device)
		with torch.no_grad():
			network_result = self.actor_net(observation)
			mean, std = self.transform_network_output(1, network_result)
			if verbose:
				v_estimate = self.critic_net(observation).view(-1)

		if mean_only:
			return mean.cpu().numpy().reshape(-1).tolist()

		action = torch.round(torch.normal(mean, std).to(self.device))
		action = torch.max(action, torch.zeros(action.shape).to(self.device))
		action = torch.min(action, 9 * torch.ones(action.shape).to(self.device))
		action = action.squeeze().type(torch.LongTensor).to('cpu').numpy()
		action = action if raw_action else self.agent_output_to_market_form(action)

		if verbose:
			transformed_network_output = np.array([mean.to('cpu').numpy(), std.to('cpu').numpy()]).reshape(-1)
			return action, transformed_network_output, v_estimate.to('cpu').item()
		else:
			return action

	def log_probability_given_action(self, states, actions):
		network_result = self.actor_net(states)
		mean, std = self.transform_network_output(network_result.shape[0], network_result)
		return torch.distributions.Normal(mean, std).log_prob(actions.view(len(mean), -1)).sum(dim=1).unsqueeze(-1)

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
		return 50000 * torch.nn.MSELoss()(proposed_actions, 3.5 * torch.ones(proposed_actions.shape).to(self.device))

	def agent_output_to_market_form(self, action):
		return action.tolist()


class ContinuosActorCriticAgentFixedOneStd(ContinuosActorCriticAgent):
	def transform_network_output(self, number_outputs, network_result):
		"""
		This implementation of transform_network_output uses the full output as mean.
		The usage of softplus ensures that the means will be non-negative.
		The standard deviation contains ones of the same shape as mean.

		Args:
			number_outputs (int): the number of independent network outputs
			network_result (torch.Tensor): The output of your network. It will be used as your mean.

		Returns:
			torch.Tensor, torch.Tensor: A tensor of your means and a tensor of your standard deviations
		"""
		network_result = network_result.view(number_outputs, -1)
		network_result = self.softplus(network_result)
		return network_result, torch.ones(network_result.shape).to(self.device)


class ContinuosActorCriticAgentEstimatingStd(ContinuosActorCriticAgent):
	def initialize_models_and_optimizer(self, n_observations, n_actions, network_architecture):
		super().initialize_models_and_optimizer(n_observations, 2 * n_actions, network_architecture)

	def transform_network_output(self, number_outputs, network_result):
		"""
		This implementation of transform_network_output splits the output into means and standard deviations.
		The usage of softplus ensures that the means and standard deviations will be non-negative.
		It is ensured that the standard deviations will be slightly greater that zero.
		To ensure that high standard deviations will be estimated with care the root of the value will be taken.

		Args:
			number_outputs (int): the number of independent network outputs
			network_result (torch.Tensor): The output of your network. It will be split into means and standard deviations.

		Returns:
			torch.Tensor, torch.Tensor: A tensor of your means and a tensor of your standard deviations
		"""
		network_result = network_result.view(number_outputs, 2, -1)
		network_result = self.softplus(network_result)
		mean = network_result[:, 0, :]
		std = network_result[:, 1, :]
		std = torch.max(std, 0.001 * torch.ones(std.shape).to(self.device))
		mean = torch.min(mean, 9 * torch.ones(mean.shape).to(self.device))
		std = torch.sqrt(std)
		return mean, std
