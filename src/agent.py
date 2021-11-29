import collections
import os
import random

import numpy as np
import torch

import model
import utils as ut
from customer import CustomerCircular
from experience_buffer import ExperienceBuffer


class Agent:
	def __init__(self):
		pass

	def policy(self, state, epsilon=0):
		assert False


class HumanPlayer(Agent):
	def __init__(self):
		print('Welcome to this funny game! Now, you are the one playing the game!')

	def policy(self, state, epsilon=0) -> int:
		print('The state is ', state, 'and you have to decide what to do! Please enter your action!')
		return int(input())


class FixedPriceAgent(Agent):
	def __init__(self, fixed_price=42):
		self.fixed_price = fixed_price

	def policy(self, state, epsilon=0) -> int:
		return self.fixed_price


class RuleBasedCEAgent(Agent):

	def __init__(self):
		pass

	def action_to_array(self, action) -> np.array:
		return [int(np.floor(action / ut.MAX_PRICE)), int(action % ut.MAX_PRICE)]

	def array_to_action(self, array) -> int:
		return array[0] * 10 + array[1]

	def policy(self, state, epsilon=0) -> int:
		# state[0]: products in my storage
		# state[1]: products in circulation
		return self.storage_evaluation(state)

	def storage_evaluation(self, state) -> int:
		# this policy sets the prices according to the amount of available storage
		products_in_storage = state[0]
		price_old = 0
		price_new = ut.PRODUCTION_PRICE
		if products_in_storage < ut.MAX_STORAGE / 4:
			# less than 1/4 of storage filled
			price_old = int(ut.MAX_PRICE * 6 / 10)
			price_new += int(ut.MAX_PRICE * 6 / 10)

		elif products_in_storage < ut.MAX_STORAGE / 2:
			# less than 1/2 of storage filled
			price_old = int(ut.MAX_PRICE * 5 / 10)
			price_new += int(ut.MAX_PRICE * 5 / 10)

		elif products_in_storage < ut.MAX_STORAGE * 3 / 4:
			# less than 3/4 but more than 1/2 of storage filled
			price_old = int(ut.MAX_PRICE * 4 / 10)
			price_new += int(ut.MAX_PRICE * 4 / 10)
		else:
			# storage too full, we need to get rid of some refurbished products
			price_old = int(ut.MAX_PRICE * 2 / 10)
			price_new += int(ut.MAX_PRICE * 7 / 10)

		price_new = min(9, price_new)
		assert price_old <= price_new
		return self.array_to_action([price_old, price_new])

	def greedy_policy(self, state) -> int:
		# this policy tries to figure out the best prices for the next round by simulating customers
		# and trying each used_price, new_price combination
		# Warning, this strategy is not very good or optimal
		customers = []
		for _ in range(0, ut.NUMBER_OF_CUSTOMERS * 10):
			customers += [CustomerCircular()]

		max_profit = -9999999999999  # we have not found a better solution yet
		max_price_n = 0
		max_price_u = 0
		for p_u in range(1, ut.MAX_PRICE):
			for p_n in range(1, ut.MAX_PRICE):
				storage = state[0]
				exp_sales_new = 0
				exp_sales_old = 0
				exp_return_prod = 0
				for customer in customers:
					c_buy, c_return = customer.buy_object([p_u, p_n, state[0], state[1]])
					# c_buy: decision, whether the customer buys 1 (old product) or two (new product)
					# c_return: decision, whether the customer returns a product
					if c_return is not None:
						exp_return_prod += 1
					if c_buy == 1:
						storage -= 1
						exp_sales_old += 1
					elif c_buy == 2:
						exp_sales_new += 1
				exp_profit = (p_n * exp_sales_new + p_u * exp_sales_old) - ((storage + exp_return_prod) * 2)
				if exp_profit > max_profit:
					max_profit = exp_profit
					max_price_n = p_n
					max_price_u = p_u
		print(max_price_u, max_price_n)
		assert max_price_n > 0 and max_price_u > 0
		return self.array_to_action[max_price_u, max_price_n]


class QLearningAgent(Agent):
	Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

	# If you enter load_path, the model will be loaded. For example, if you want to use a pretrained net or test a given agent.
	# If you set an optim, this means you want training.
	# Give no optim if you don't want training.
	def __init__(self, n_observation, n_actions, optim=None, device='cpu', load_path=None):
		self.device = device
		self.n_actions = n_actions
		self.buffer_for_feedback = None
		self.optimizer = None
		print('I initiate a QLearningAgent using {} device'.format(self.device))
		self.net = model.simple_network(n_observation, n_actions).to(self.device)
		if load_path:
			self.net.load_state_dict(torch.load(load_path, map_location=self.device))
		if optim:
			self.optimizer = optim(self.net.parameters(), lr=ut.LEARNING_RATE)
			self.tgt_net = model.simple_network(n_observation, n_actions).to(self.device)
			if load_path:
				self.tgt_net.load_state_dict(torch.load(load_path), map_location=self.device)
			self.buffer = ExperienceBuffer(ut.REPLAY_SIZE)

	@torch.no_grad()
	def policy(self, state, epsilon=0):
		assert self.buffer_for_feedback is None or self.optimizer is None
		if np.random.random() < epsilon:
			action = random.randint(0, self.n_actions - 1)
		else:
			action = int(torch.argmax(self.net(torch.Tensor(state).to(self.device))))
		if self.optimizer is not None:
			self.buffer_for_feedback = (state, action)
		return action

	def set_feedback(self, reward, is_done, new_state):
		exp = self.Experience(*self.buffer_for_feedback, reward, is_done, new_state)
		self.buffer.append(exp)
		self.buffer_for_feedback = None

	def train_batch(self):
		self.optimizer.zero_grad()
		batch = self.buffer.sample(ut.BATCH_SIZE)
		loss_t, selected_q_val_mean = self.calc_loss(batch)
		loss_t.backward()
		self.optimizer.step()
		return loss_t.item(), selected_q_val_mean.item()

	def synchronize_tgt_net(self):
		self.tgt_net.load_state_dict(self.net.state_dict())

	def calc_loss(self, batch, device='cpu'):
		states, actions, rewards, dones, next_states = batch

		states_v = torch.tensor(np.single(states)).to(device)
		next_states_v = torch.tensor(np.single(next_states)).to(device)
		actions_v = torch.tensor(actions).to(device)
		rewards_v = torch.tensor(rewards).to(device)
		done_mask = torch.BoolTensor(dones).to(device)

		state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

		with torch.no_grad():
			next_state_values = self.tgt_net(next_states_v).max(1)[0]
			next_state_values[done_mask] = 0.0
			next_state_values = next_state_values.detach()

		expected_state_action_values = next_state_values * ut.GAMMA + rewards_v
		return torch.nn.MSELoss()(state_action_values, expected_state_action_values), state_action_values.mean()

	def save(self, path='QLearning_parameters'):
		if not os.path.isdir('trainedModels'):
			os.mkdir('trainedModels')
		torch.save(self.net.state_dict(), './trainedModels/' + path + '.dat')
