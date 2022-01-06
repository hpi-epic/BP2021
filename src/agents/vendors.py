import collections
import math
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch

import configuration.utils_rl as ut_rl
import configuration.utils_sim_market as ut
import rl.model as model
from market.customer import CustomerCircular
from rl.experience_buffer import ExperienceBuffer


class Agent(ABC):

	def __init__(self, name='agent'):
		self.name = name

	def custom_init(self, class_name, args):
		"""
		Initialize an agent with a list of arguments.

		Args:
			class_name (agent class): The class of the agent that should be instantiated.
			args (list): List of arguments to pass the initializer.

		Returns:
			agent instance: An instance of the agent_class initialized with the given args.
		"""
		return class_name(*args)

	@abstractmethod
	def policy(self, observation, epsilon=0):
		raise NotImplementedError


class CircularAgent(Agent, ABC):
	pass


class LinearAgent(Agent, ABC):
	pass


class RuleBasedAgent(Agent, ABC):
	pass


class ReinforcementLearningAgent(Agent, ABC):
	pass


class HumanPlayer(RuleBasedAgent, ABC):
	@abstractmethod
	def policy(self, observation, *_) -> int:
		raise NotImplementedError


class HumanPlayerLE(LinearAgent, HumanPlayer):
	def __init__(self, name='YOU - Linear'):
		self.name = name
		print('Welcome to this funny game! Now, you are the one playing the game!')

	def policy(self, observation, *_) -> int:
		print('The observation is', observation, 'and you have to decide what to do! Please enter your actions, seperated by spaces!')
		return input()


class HumanPlayerCE(CircularAgent, HumanPlayer):
	def __init__(self, name='YOU - Circular'):
		self.name = name
		print('Welcome to this funny game! Now, you are the one playing the game!')

	def policy(self, observation, *_) -> int:
		raw_input_string = super().policy(observation)
		assert raw_input_string.count(' ') == 1, 'Please enter two numbers seperated by spaces!'
		price_old, price_new = raw_input_string.split(' ')
		return (int(price_old), int(price_new))


class HumanPlayerCERebuy(HumanPlayerCE):
	def policy(self, observation, *_) -> int:
		raw_input_string = super().policy(observation)
		assert raw_input_string.count(' ') == 2, 'Please enter three numbers seperated by spaces!'
		price_old, price_new, rebuy_price = raw_input_string.split(' ')
		return (int(price_old), int(price_new), int(rebuy_price))


class FixedPriceAgent(RuleBasedAgent, ABC):
	"""
	An abstract class for FixedPriceAgents
	"""
	pass


class FixedPriceLEAgent(LinearAgent, FixedPriceAgent):
	def __init__(self, fixed_price=ut.PRODUCTION_PRICE + 3, name='fixed_price_le'):
		assert isinstance(fixed_price, int), 'The fixed_price must be an integer'
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> int:
		return self.fixed_price


class FixedPriceCEAgent(CircularAgent, FixedPriceAgent):
	def __init__(self, fixed_price=(2, 4), name='fixed_price_ce'):
		assert isinstance(fixed_price, tuple) and len(fixed_price) == 2, 'The fixed_price must be a tuple of integers'
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> int:
		return self.fixed_price


class FixedPriceCERebuyAgent(FixedPriceCEAgent):
	def __init__(self, fixed_price=(3, 6, 2), name='fixed_price_ce_rebuy'):
		assert isinstance(fixed_price, tuple) and len(fixed_price) == 3, 'The fixed_price must be a triple of integers'
		self.name = name
		self.fixed_price = fixed_price

	def policy(self, *_) -> int:
		return self.fixed_price


class RuleBasedCEAgent(RuleBasedAgent, CircularAgent):
	def __init__(self, name='rule_based_ce'):
		self.name = name

	def return_prices(self, price_old, price_new, rebuy_price):
		return (price_old, price_new)

	def policy(self, observation, epsilon=0) -> int:
		# this policy sets the prices according to the amount of available storage
		products_in_storage = observation[1]
		price_old = 0
		price_new = ut.PRODUCTION_PRICE
		rebuy_price = 0
		if products_in_storage < ut.MAX_STORAGE / 15:
			# fill up the storage immediately
			price_old = int(ut.MAX_PRICE * 6 / 10)
			price_new += int(ut.MAX_PRICE * 6 / 10)
			rebuy_price = price_old - 1

		elif products_in_storage < ut.MAX_STORAGE / 10:
			# fill up the storage
			price_old = int(ut.MAX_PRICE * 5 / 10)
			price_new += int(ut.MAX_PRICE * 5 / 10)
			rebuy_price = price_old - 2

		elif products_in_storage < ut.MAX_STORAGE / 8:
			# storage content is ok
			price_old = int(ut.MAX_PRICE * 4 / 10)
			price_new += int(ut.MAX_PRICE * 4 / 10)
			rebuy_price = int(price_old / 2)
		else:
			# storage too full, we need to get rid of some refurbished products
			price_old = int(ut.MAX_PRICE * 2 / 10)
			price_new += int(ut.MAX_PRICE * 7 / 10)
			rebuy_price = 0

		price_new = min(9, price_new)
		assert price_old <= price_new
		return self.return_prices(price_old, price_new, rebuy_price)

	def greedy_policy(self, observation) -> int:
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
				storage = observation[0]
				exp_sales_new = 0
				exp_sales_old = 0
				exp_return_prod = 0
				for customer in customers:
					c_buy, c_return = customer.buy_object([p_u, p_n, observation[0], observation[1]])
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
		return (max_price_u, max_price_n)


class RuleBasedCERebuyAgent(RuleBasedCEAgent):
	def return_prices(self, price_old, price_new, rebuy_price):
		return (price_old, price_new, rebuy_price)


class QLearningAgent(ReinforcementLearningAgent, ABC):
	Experience = collections.namedtuple('Experience', field_names=['observation', 'action', 'reward', 'done', 'new_observation'])

	# If you enter load_path, the model will be loaded. For example, if you want to use a pretrained net or test a given agent.
	# If you set an optim, this means you want training.
	# Give no optim if you don't want training.
	def __init__(self, n_observation, n_actions, optim=None, device='cuda' if torch.cuda.is_available() else 'cpu', load_path=None, name='q_learning'):
		self.device = device
		self.n_actions = n_actions
		self.buffer_for_feedback = None
		self.optimizer = None
		self.name = name
		# print(f'I initiate a QLearningAgent using {self.device} device')
		self.net = model.simple_network(n_observation, n_actions).to(self.device)
		if load_path:
			self.net.load_state_dict(torch.load(load_path, map_location=self.device))
		if optim:
			self.optimizer = optim(self.net.parameters(), lr=ut_rl.LEARNING_RATE)
			self.tgt_net = model.simple_network(n_observation, n_actions).to(self.device)
			if load_path:
				self.tgt_net.load_state_dict(torch.load(load_path), map_location=self.device)
			self.buffer = ExperienceBuffer(ut_rl.REPLAY_SIZE)

	@torch.no_grad()
	def policy(self, observation, epsilon=0):
		assert self.buffer_for_feedback is None or self.optimizer is None
		if np.random.random() < epsilon:
			action = random.randint(0, self.n_actions - 1)
		else:
			action = int(torch.argmax(self.net(torch.Tensor(observation).to(self.device))))
		if self.optimizer is not None:
			self.buffer_for_feedback = (observation, action)
		return action

	def set_feedback(self, reward, is_done, new_observation):
		exp = self.Experience(*self.buffer_for_feedback, reward, is_done, new_observation)
		self.buffer.append(exp)
		self.buffer_for_feedback = None

	def train_batch(self):
		self.optimizer.zero_grad()
		batch = self.buffer.sample(ut_rl.BATCH_SIZE)
		loss_t, selected_q_val_mean = self.calc_loss(batch, self.device)
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

		expected_state_action_values = next_state_values * ut_rl.GAMMA + rewards_v
		return torch.nn.MSELoss()(state_action_values, expected_state_action_values), state_action_values.mean()

	def save(self, path='QLearning_parameters'):
		if not os.path.isdir('trainedModels'):
			os.mkdir('trainedModels')
		torch.save(self.net.state_dict(), './trainedModels/' + path)


class QLearningLEAgent(QLearningAgent, LinearAgent):
	pass


class QLearningCEAgent(QLearningAgent, CircularAgent):
	def policy(self, observation, epsilon=0) -> int:
		step = super().policy(observation, epsilon)
		return (int(step % 10), int(step / 10))


class QLearningCERebuyAgent(QLearningAgent, CircularAgent):
	def policy(self, observation, epsilon=0) -> int:
		step = super().policy(observation, epsilon)
		return (int(step / 100), int(step / 10 % 10), int(step % 10))


class CompetitorLinearRatio1(LinearAgent, RuleBasedAgent):
	def policy(self, state, epsilon=0):
		# this stratgy calculates the value per money for each competing vendor and tries to adapt to it
		ratios = []
		# ratios[0] is the ratio of the competitor itself. it is compared to the other ratios
		max_competing_ratio = 0
		for i in range(math.floor(len(state) / 2)):
			quality_opponent = state[2 * i + 2]
			price_opponent = state[2 * i + 1] + 1
			ratio = quality_opponent / price_opponent  # value for price for vendor i
			ratios.append(ratio)
			if ratio > max_competing_ratio:
				max_competing_ratio = ratio

		ratio = max_competing_ratio / ratios[0]
		intended = math.floor(1 / max_competing_ratio * state[0]) - 1
		actual_price = min(max(ut.PRODUCTION_PRICE + 1, intended), ut.MAX_PRICE - 1)
		# print('price from the competitor:', actual_price)
		return actual_price


class CompetitorRandom(LinearAgent, RuleBasedAgent):
	def policy(self, state, epsilon=0):
		return random.randint(ut.PRODUCTION_PRICE + 1, ut.MAX_PRICE - 1)


class CompetitorJust2Players(LinearAgent, RuleBasedAgent):
	def policy(self, state, epsilon=0) -> int:
		"""	This competitor is based on quality and agents actions.
		While he can act in every linear economy, you should not expect good performance in a multicompetitor setting.

		Args:
			state (np.array): The state of the marketplace the agent sells its products at.
			epsilon (int, optional): Not used it this method. Defaults to 0.

		Returns:
			int: The price of the product he sells in the next round.
		"""
		# assert len(state) == 4, "You can't use this competitor in this market!"
		agent_price = state[1]
		agent_quality = state[2]
		comp_quality = state[0]

		new_price = 0

		if comp_quality > agent_quality + 15:
			# significantly better quality
			new_price = agent_price + 2
		elif comp_quality > agent_quality:
			# slightly better quality
			new_price = agent_price + 1
		elif comp_quality < agent_quality and comp_quality > agent_quality - 15:
			# slightly worse quality
			new_price = agent_price - 1
		elif comp_quality < agent_quality:
			# significantly worse quality
			new_price = agent_price - 2
		elif comp_quality == agent_quality:
			# same quality
			new_price = agent_price
		if new_price < ut.PRODUCTION_PRICE:
			new_price = ut.PRODUCTION_PRICE + 1
		elif new_price >= ut.MAX_PRICE:
			new_price = ut.MAX_PRICE - 1
		new_price = int(new_price)
		assert isinstance(new_price, int)
		return new_price
