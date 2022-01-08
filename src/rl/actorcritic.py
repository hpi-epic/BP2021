import random
import time

import model
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import agents.vendors as vendors
import configuration.utils_rl as ut_rl
import configuration.utils_sim_market as ut
import market.sim_market as sim_market


class ActorCriticAgent(vendors.Agent):
	def __init__(self, n_observation, n_actions):
		self.device = 'cpu'
		self.policy_net = model.simple_network(n_observation, n_actions).to(self.device)
		self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0000025)
		self.v_net = model.simple_network(n_observation, 1).to(self.device)
		self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=0.00025)

	def train_batch(self, states, actions, rewards, states_dash):
		print('Diese Mittelwerte hat das Netz vor dem Training vorgeschlagen:')
		print(self.policy_net(states))
		print('Das waren die Aktionen: ')
		print(actions)
		print('Das waren die Rewards: ')
		print(rewards)
		states = states.to(self.device)
		actions = actions.to(self.device)
		rewards = rewards.to(self.device)
		states_dash = states_dash.to(self.device)
		self.v_optimizer.zero_grad()
		self.policy_optimizer.zero_grad()

		v_estimates = self.v_net(states)
		with torch.no_grad():
			v_expected = (rewards + ut_rl.GAMMA * self.v_net(states_dash).detach()).view(-1, 1)
		valueloss = torch.nn.MSELoss()(v_estimates, v_expected)
		valueloss.backward()

		with torch.no_grad():
			baseline = 0  # v_estimates.squeeze()[31].item()
			constant = (v_expected - baseline).detach()
		# print("Seine Konstante war", constant)
		log_policy = torch.log(self.probability_given_action(states, actions))
		policyloss = torch.sum(constant * log_policy)
		policyloss.backward()

		self.v_optimizer.step()
		self.policy_optimizer.step()
		print('Diese Mittelwerte schlägt das Netz nach dem Training vor:')
		print(self.policy_net(states))
		print('\n\n\n')

		return valueloss.to('cpu').item(), policyloss.to('cpu').item()


class DiscreteActorCriticAgent(ActorCriticAgent):
	def policy(self, observation):
		observation = torch.Tensor(observation).to(self.device)
		with torch.no_grad():
			distribution = torch.softmax(self.policy_net(observation).view(-1), dim=0)
			v_estimat = self.v_net(observation).view(-1)

		distribution = distribution.to('cpu').detach().numpy()
		action = ut.shuffle_from_probabilities(distribution)
		return action, distribution[action], v_estimat.to('cpu').item()

	def probability_given_action(self, states, actions):
		return torch.softmax(self.policy_net(states), dim=0).gather(1, actions.unsqueeze(-1))


class SoftActorCriticAgent(ActorCriticAgent):
	def policy(self, observation):
		observation = torch.Tensor(observation).to(self.device)
		with torch.no_grad():
			mean = self.policy_net(observation)
			v_estimat = self.v_net(observation).view(-1)

		action = torch.normal(mean, torch.ones(mean.shape)).to('cpu').item()
		action = max(action, 0)
		action = min(action, 9)
		return action, v_estimat.to('cpu').item()

	def probability_given_action(self, states, actions):
		return torch.exp(-torch.square(actions - self.policy_net(states))) / (torch.sqrt(2 * np.pi * torch.ones(states.shape())))


def trainactorcritic():
	agent = SoftActorCriticAgent(6, 1)
	all_dicts = []
	all_probs = []
	all_vestim = []
	all_value_losses = []
	all_policy_losses = []
	writer = SummaryWriter(log_dir='runs/' + time.strftime('%Y%m%d-%H%M%S'))

	episodes_accomplished = 0
	total_envs = 128
	environments = [sim_market.ClassicScenario() for _ in range(total_envs)]
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
			step, prob, v_estimat = agent.policy(state)
			all_probs.append(prob)
			all_vestim.append(v_estimat)
			state_dash, reward, isdone, info = environments[env].step((int(step / 100), int(step / 10 % 10), int(step % 10)))

			states.append(state)
			actions.append(step)
			rewards.append(reward)
			states_dash.append(state_dash)
			# print(state)
			# print(state_dash)
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
				writer.add_scalar('training/prob_mean', np.mean(all_probs[-1000:]), episodes_accomplished)
				writer.add_scalar('training/v_estim', np.mean(all_vestim[-1000:]), episodes_accomplished)
				writer.add_scalar('loss/value', np.mean(all_value_losses[-1000:]), episodes_accomplished)
				writer.add_scalar('loss/policy', np.mean(all_policy_losses[-1000:]), episodes_accomplished)

				environments[env].reset()
				info_accumulators[env] = None

		valueloss, policyloss = agent.train_batch(torch.Tensor(np.array(states)), torch.from_numpy(np.array(actions, dtype=np.int64)), torch.Tensor(np.array(rewards)), torch.Tensor(np.array(state_dash)))
		all_value_losses.append(valueloss)
		all_policy_losses.append(policyloss)


trainactorcritic()
