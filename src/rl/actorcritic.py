import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut
import market.sim_market as sim_market
import rl.model as model


class ActorCriticAgent(vendors.Agent):
	def __init__(self, n_observations, n_actions):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		print(f'I initiate an ActorCriticAgent using {self.device} device')
		self.initialize_models_and_optimizer(n_observations, n_actions)

	def train_batch(self, states, actions, rewards, states_dash, regularization=False):
		states = states.to(self.device)
		actions = actions.to(self.device)
		rewards = rewards.to(self.device)
		states_dash = states_dash.to(self.device)
		self.v_optimizer.zero_grad()
		self.policy_optimizer.zero_grad()

		v_estimates = self.v_net(states)
		with torch.no_grad():
			v_expected = (rewards + config.GAMMA * self.v_net(states_dash).detach()).view(-1, 1)
		valueloss = torch.nn.MSELoss()(v_estimates, v_expected)
		valueloss.backward()

		with torch.no_grad():
			baseline = v_estimates.squeeze()[31].item()
			constant = (v_expected - baseline).detach()
		log_prob = -self.log_probability_given_action(states.detach(), actions.detach())
		policy_loss = torch.mean(constant * log_prob)
		if regularization:
			policy_loss += self.regularizate(states, actions)
		policy_loss.backward()

		self.v_optimizer.step()
		self.policy_optimizer.step()

		return valueloss.to('cpu').item(), policy_loss.to('cpu').item()

	def regularizate(self, *_):
		return 0


class DiscreteActorCriticAgent(ActorCriticAgent):
	def initialize_models_and_optimizer(self, n_observations, n_actions):
		self.policy_net = model.simple_network(n_observations, n_actions).to(self.device)
		self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0000025)
		self.v_net = model.simple_network(n_observations, 1).to(self.device)
		self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=0.00025)

	def policy(self, observation, verbose=False):
		observation = torch.Tensor(observation).to(self.device)
		with torch.no_grad():
			distribution = torch.softmax(self.policy_net(observation).view(-1), dim=0)
			if verbose:
				v_estimat = self.v_net(observation).view(-1)

		distribution = distribution.to('cpu').detach().numpy()
		action = ut.shuffle_from_probabilities(distribution)
		return action, distribution[action], v_estimat.to('cpu').item() if verbose else None

	def log_probability_given_action(self, states, actions):
		return -torch.log(torch.softmax(self.policy_net(states), dim=0).gather(1, actions.unsqueeze(-1)))


class DiscreteACALinear(DiscreteActorCriticAgent):
	def agent_output_to_market_form(self, step):
		return step


class DiscreteACACircularEconomy(DiscreteActorCriticAgent):
	def agent_output_to_market_form(self, step):
		return (int(step % 10), int(step / 10))


class DiscreteACACircularEconomyRebuy(DiscreteActorCriticAgent):
	def agent_output_to_market_form(self, step):
		return (int(step / 100), int(step / 10 % 10), int(step % 10))


class SoftActorCriticAgent(ActorCriticAgent):
	softplus = torch.nn.Softplus()

	def initialize_models_and_optimizer(self, n_observations, n_actions):
		self.n_actions = n_actions
		self.policy_net = model.simple_network(n_observations, self.n_actions).to(self.device)
		self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0002)
		self.v_net = model.simple_network(n_observations, 1).to(self.device)
		self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=0.002)

	def policy(self, observation, verbose=False):
		observation = torch.Tensor([observation]).to(self.device)
		with torch.no_grad():
			mean = self.softplus(self.policy_net(observation))
			if verbose:
				v_estimat = self.v_net(observation).view(-1)

		action = torch.round(torch.normal(mean, torch.ones(mean.shape)))
		action = torch.max(action, torch.zeros(action.shape))
		action = torch.min(action, 9 * torch.ones(action.shape))
		return action.squeeze().type(torch.LongTensor).to('cpu').numpy(), *((self.log_probability_given_action(observation, action).mean().detach().item(), v_estimat.to('cpu').item()) if verbose else (None, None))

	def log_probability_given_action(self, states, actions):
		return torch.distributions.Normal(self.softplus(self.policy_net(states)), 1).log_prob(actions.view(-1, self.n_actions)).sum(dim=1).unsqueeze(-1)

	def regularizate(self, states, actions):
		return 50000 * torch.nn.MSELoss()(self.policy_net(states), 3.5 * torch.ones(actions.shape))

	def agent_output_to_market_form(self, step):
		return step.tolist()


def train_actorcritic(Scenario=sim_market.CircularEconomyRebuyPriceOneCompetitor, Agent=SoftActorCriticAgent, outputs=3, verbose=False):
	agent = Agent(Scenario().observation_space.shape[0], outputs)
	assert isinstance(agent, ActorCriticAgent)
	all_dicts = []
	if verbose:
		all_probs = []
		all_v_estimates = []
	all_value_losses = []
	all_policy_losses = []
	writer = SummaryWriter(log_dir='runs/' + time.strftime('%Y%m%d-%H%M%S'))

	episodes_accomplished = 0
	total_envs = 128
	environments = [Scenario() for _ in range(total_envs)]
	info_accumulators = [None for _ in range(total_envs)]
	for i in range(10000):
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
			step, prob, v_estimate = agent.policy(state, verbose)
			if verbose:
				all_probs.append(prob)
				all_v_estimates.append(v_estimate)
			state_dash, reward, isdone, info = environments[env].step(agent.agent_output_to_market_form(step))

			states.append(state)
			actions.append(step)
			rewards.append(reward)
			states_dash.append(state_dash)
			info_accumulators[env] = info if info_accumulators[env] is None else ut.add_content_of_two_dicts(info_accumulators[env], info)

			if isdone:
				episodes_accomplished += 1
				if episodes_accomplished % 10 == 0:
					print('I accomplished', episodes_accomplished, 'episodes')
				all_dicts.append(info_accumulators[env])

				# calculate the average of the last 100 items
				sliced_dicts = all_dicts[-100:]
				averaged_info = sliced_dicts[0]
				for i, next_dict in enumerate(sliced_dicts):
					if i != 0:
						averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
				averaged_info = ut.divide_content_of_dict(averaged_info, len(sliced_dicts))
				ut.write_dict_to_tensorboard(writer, averaged_info, episodes_accomplished, is_cumulative=True)
				if verbose:
					writer.add_scalar('training/prob_mean', np.mean(all_probs[-1000:]), episodes_accomplished)
					writer.add_scalar('training/v_estimate', np.mean(all_v_estimates[-1000:]), episodes_accomplished)
				writer.add_scalar('loss/value', np.mean(all_value_losses[-1000:]), episodes_accomplished)
				writer.add_scalar('loss/policy', np.mean(all_policy_losses[-1000:]), episodes_accomplished)

				environments[env].reset()
				info_accumulators[env] = None

		valueloss, policy_loss = agent.train_batch(torch.Tensor(np.array(states)), torch.from_numpy(np.array(actions, dtype=np.int64)), torch.Tensor(np.array(rewards)), torch.Tensor(np.array(state_dash)), episodes_accomplished <= 500)
		all_value_losses.append(valueloss)
		all_policy_losses.append(policy_loss)


if __name__ == '__main__':
	train_actorcritic(Agent=DiscreteACACircularEconomyRebuy, outputs=1000, verbose=True)
