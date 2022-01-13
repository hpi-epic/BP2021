import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut
import market.sim_market as sim_market
import rl.model as model


class ActorCriticAgent(vendors.Agent, ABC):
	"""
	This is an implementation of an (one step) actor critic agent as proposed in Richard Suttons textbook on page 332.
	"""
	def __init__(self, n_observations, n_actions):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		print(f'I initiate an ActorCriticAgent using {self.device} device')
		self.initialize_models_and_optimizer(n_observations, n_actions)

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
			v_expected = (rewards + config.GAMMA * self.critic_net(states_dash).detach()).view(-1, 1)
		valueloss = torch.nn.MSELoss()(v_estimates, v_expected)
		valueloss.backward()

		with torch.no_grad():
			baseline = v_estimates.squeeze()[31].item()
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
			torch.Tensor or 0: The punishment for the agent
		"""
		return 0

	@abstractmethod
	def log_probability_given_action(self, states, actions) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def agent_output_to_market_form(self) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')


class DiscreteActorCriticAgent(ActorCriticAgent):
	"""
	This is an actor critic agent with continuos action space.
	It generates preferences and uses softmax to gain the probabilities.
	For our three markets we have three kinds of specific agents you must use.
	"""
	def initialize_models_and_optimizer(self, n_observations, n_actions):
		self.actor_net = model.simple_network(n_observations, n_actions).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0000025)
		self.critic_net = model.simple_network(n_observations, 1).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=0.00025)

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
		return (int(action % 10), int(action / 10))


class DiscreteACACircularEconomyRebuy(DiscreteActorCriticAgent):
	def agent_output_to_market_form(self, action):
		return (int(action / 100), int(action / 10 % 10), int(action % 10))


class ContinuosActorCriticAgent(ActorCriticAgent):
	"""
	This is an actor critic agent with continuos action space.
	It's parametrization is a normal distribution with fixed mean.
	It works on any sort of market we have so far, just the number of action values must be given.
	"""
	softplus = torch.nn.Softplus()

	def initialize_models_and_optimizer(self, n_observations, n_actions):
		self.n_actions = n_actions
		self.actor_net = model.simple_network(n_observations, self.n_actions).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=0.0002)
		self.critic_net = model.simple_network(n_observations, 1).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=0.002)

	def policy(self, observation, verbose=False):
		observation = torch.Tensor([observation]).to(self.device)
		with torch.no_grad():
			mean = self.softplus(self.actor_net(observation))
			if verbose:
				v_estimat = self.critic_net(observation).view(-1)

		action = torch.round(torch.normal(mean, torch.ones(mean.shape)))
		action = torch.max(action, torch.zeros(action.shape))
		action = torch.min(action, 9 * torch.ones(action.shape))
		return action.squeeze().type(torch.LongTensor).to('cpu').numpy(), *((self.log_probability_given_action(observation, action).mean().detach().item(), v_estimat.to('cpu').item()) if verbose else (None, None))

	def log_probability_given_action(self, states, actions):
		return torch.distributions.Normal(self.softplus(self.actor_net(states)), 1).log_prob(actions.view(-1, self.n_actions)).sum(dim=1).unsqueeze(-1)

	def regularize(self, states):
		"""
		This regularization pushes the actor with very high priority towards a mean price of 3.5.
		Use it at the beginning to avoid 0 pricing which gets only horrible negative reward.

		Args:
			states (torch.Tensor): The current states the agent is in at the moment

		Returns:
			torch.Tensor: the malus of the regularization
		"""
		proposed_actions = self.actor_net(states.detach())
		return 50000 * torch.nn.MSELoss()(proposed_actions, 3.5 * torch.ones(proposed_actions.shape))

	def agent_output_to_market_form(self, action):
		return action.tolist()


def train_actorcritic(marketplace_class=sim_market.CircularEconomyRebuyPriceOneCompetitor, agent_class=ContinuosActorCriticAgent, outputs=3, number_of_training_steps=1000, verbose=False):
	assert issubclass(agent_class, ActorCriticAgent), f'the agent_class must be a subclass of ActorCriticAgent: {agent_class}'
	agent = agent_class(marketplace_class().observation_space.shape[0], outputs)

	all_dicts = []
	if verbose:
		all_probs = []
		all_v_estimates = []
	all_value_losses = []
	all_policy_losses = []
	writer = SummaryWriter()

	finished_episodes = 0
	total_envs = 128
	environments = [marketplace_class() for _ in range(total_envs)]
	info_accumulators = [None for _ in range(total_envs)]
	for i in range(number_of_training_steps):
		# choose 32 environments
		chosen_envs = set()
		# chosen_envs.add(127)
		while len(chosen_envs) < 32:
			number = random.randint(0, total_envs - 1)
			if number not in chosen_envs:
				chosen_envs.add(number)

		states = []
		actions = []
		rewards = []
		states_dash = []
		for env in chosen_envs:
			state = environments[env].observation()
			action, prob, v_estimate = agent.policy(state, verbose)
			if verbose:
				all_probs.append(prob)
				all_v_estimates.append(v_estimate)
			state_dash, reward, is_done, info = environments[env].step(agent.agent_output_to_market_form(action))

			states.append(state)
			actions.append(action)
			rewards.append(reward)
			states_dash.append(state_dash)
			info_accumulators[env] = info if info_accumulators[env] is None else ut.add_content_of_two_dicts(info_accumulators[env], info)

			if is_done:
				finished_episodes += 1
				if finished_episodes % 10 == 0:
					print(f'Finished {finished_episodes} episodes')
				all_dicts.append(info_accumulators[env])

				# calculate the average of the last 100 items
				sliced_dicts = all_dicts[-100:]
				averaged_info = sliced_dicts[0]
				for i, next_dict in enumerate(sliced_dicts):
					if i != 0:
						averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
				averaged_info = ut.divide_content_of_dict(averaged_info, len(sliced_dicts))
				ut.write_dict_to_tensorboard(writer, averaged_info, finished_episodes, is_cumulative=True)
				if verbose:
					writer.add_scalar('training/prob_mean', np.mean(all_probs[-1000:]), finished_episodes)
					writer.add_scalar('training/v_estimate', np.mean(all_v_estimates[-1000:]), finished_episodes)
				writer.add_scalar('loss/value', np.mean(all_value_losses[-1000:]), finished_episodes)
				writer.add_scalar('loss/policy', np.mean(all_policy_losses[-1000:]), finished_episodes)

				environments[env].reset()
				info_accumulators[env] = None

		valueloss, policy_loss = agent.train_batch(torch.Tensor(np.array(states)), torch.from_numpy(np.array(actions, dtype=np.int64)), torch.Tensor(np.array(rewards)), torch.Tensor(np.array(state_dash)), finished_episodes <= 500)
		all_value_losses.append(valueloss)
		all_policy_losses.append(policy_loss)


if __name__ == '__main__':
	train_actorcritic()
