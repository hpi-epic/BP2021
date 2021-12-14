#!/usr/bin/env python3
import copy
from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np

import competitor as comp
import customer
import owner
import utils as ut
from customer import Customer
from owner import Owner

# An offer is a Market State that contains both prices and both qualities

# There are three kinds of state:
# First: a common state for all vendors
# Second: a state specific to one vendor
# Third: vendor's actions from the former round which needs to be saved and influence the other's decision e.g. prices

class SimMarket(gym.Env, ABC):
	def __init__(self) -> None:
		self.competitors = self.get_competitor_list()
		# The agent's price does not belong to the observation_space any more because an agent should not depend on it
		self.setup_action_observation_space()
		self.owner = None
		self.customer = None
		# TODO: Better testing for the observation and action space
		assert (
			self.observation_space and self.action_space
		), 'Your subclass has major problems with setting up the environment'

	# The number of competitors plus the agent
	def n_vendors(self) -> int:
		return len(self.competitors) + 1

	def reset(self) -> np.array:
		self.step_counter = 0

		self.set_common_state()

		self.vendor_specific_state = []
		for _ in range(self.n_vendors()):
			self.vendor_specific_state.append(self.reset_specific_vendor_state())

		self.vendors_actions = []
		for _ in range(self.n_vendors()):
			self.vendors_actions.append(ut.PRODUCTION_PRICE + 1)

		self.customer = self.choose_customer()
		self.owner = self.choose_owner()

		return self.observation()

	def simulate_customers(self, profits, offers, n) -> None:
		self.customer.set_probabilities_from_offers(offers)
		for _ in range(n):
			customer_buy = self.customer.buy_object(offers)
			if customer_buy != 0:
				self.complete_purchase(offers, profits, customer_buy)

	def step(self, action) -> Tuple[np.array, np.float64, bool, dict]:
		# The action is the new price of the agent

		err_msg = '%r (%s) invalid' % (action, type(action))
		assert self.action_space.contains(action), err_msg
		self.vendors_actions[0] = action + 1

		self.step_counter += 1

		profits = [0] * self.n_vendors()

		self.consider_owners_return(self.generate_customer_offer(), profits)

		for i in range(self.n_vendors()):
			self.simulate_customers(
				profits,
				self.generate_customer_offer(),
				int(np.floor(ut.NUMBER_OF_CUSTOMERS / self.n_vendors())),
			)
			# the competitor, which turn it is, will update its pricing
			if i < len(self.competitors):
				action_competitor_i = self.competitors[i].policy(
					self.observation(i + 1)
				)
				assert self.action_space.contains(action_competitor_i), 'The %dth vendor does not deliver a suitable action'.format(i + 1)
				self.vendors_actions[i + 1] = action_competitor_i + 1

		self.consider_storage_costs(profits)

		output_dict = {'all_profits': profits}
		is_done = self.step_counter >= ut.EPISODE_LENGTH
		return self.observation(), profits[0], is_done, output_dict

	def observation(self, vendor_view=0):
		obs = self.get_common_state_array()
		assert isinstance(obs, np.ndarray), 'get_common_state_array must return a np-Array'
		obs = np.concatenate((obs, np.array(self.vendor_specific_state[vendor_view], ndmin=1)), dtype=np.float64)
		for i in range(self.n_vendors()):
			if i == vendor_view:
				continue
			obs = np.concatenate((obs, np.array(self.vendors_actions[i], ndmin=1)), dtype=np.float64)
			obs = np.concatenate((obs, np.array(self.vendor_specific_state[i], ndmin=1)), dtype=np.float64)
		err_msg = '%r (%s) invalid observation' % (obs, type(obs))
		assert self.observation_space.contains(obs), err_msg
		return obs

	def generate_customer_offer(self):
		offer = self.get_common_state_array()
		assert isinstance(offer, np.ndarray), 'get_common_state_array must return a np-Array'
		for i in range(self.n_vendors()):
			offer = np.concatenate((offer, np.array(self.vendors_actions[i], ndmin=1)), dtype=np.float64)
			offer = np.concatenate((offer, np.array(self.vendor_specific_state[i], ndmin=1)), dtype=np.float64)
		return offer

	def set_common_state(self) -> None:
		pass
	
	def get_common_state_array(self) -> np.array:
		return np.array([])

	@abstractmethod
	def consider_owners_return(self, *_) -> None:
		raise NotImplementedError

	@abstractmethod
	def get_competitor_list(self) -> list:
		raise NotImplementedError

	@abstractmethod
	def consider_storage_costs(self, profits) -> None:
		raise NotImplementedError

	def choose_owner(self):
		pass


class LinearEconomy(SimMarket, ABC):

	def setup_action_observation_space(self) -> None:
		# cell 0: agent's quality, afterwards: odd cells: competitor's price, even cells: competitor's quality
		self.observation_space = gym.spaces.Box(
			np.array([0.0] * (len(self.competitors) * 2 + 1)),
			np.array(
				[ut.MAX_QUALITY]
				+ [ut.MAX_PRICE, ut.MAX_QUALITY] * len(self.competitors)
			),
			dtype=np.float64,
		)

		# one action for every price possible for 0 and MAX_PRICE
		self.action_space = gym.spaces.Discrete(ut.MAX_PRICE)

	def reset_specific_vendor_state(self) -> int:
		return ut.shuffle_quality()

	# def action_to_array(self, action) -> np.array:
	# 	return np.array([action + 1.0])

	def choose_customer(self) -> Customer:
		return customer.CustomerLinear()

	def complete_purchase(self, offers, profits, customer_buy) -> None:
		profits[customer_buy - 1] += (offers[(customer_buy - 1) * 2] - ut.PRODUCTION_PRICE)

	def consider_owners_return(self, offer, profits) -> None:
		pass

	def consider_storage_costs(self, profits) -> None:
		pass

	@abstractmethod
	def get_competitor_list(self) -> list:
		raise NotImplementedError


class ClassicScenario(LinearEconomy):

	def get_competitor_list(self) -> list:
		return [comp.CompetitorLinearRatio1()]


class MultiCompetitorScenario(LinearEconomy):

	def get_competitor_list(self) -> list:
		return [
			comp.CompetitorLinearRatio1(),
			comp.CompetitorRandom(),
			comp.CompetitorJust2Players(),
		]


class CircularEconomy(SimMarket):
	# currently monopoly
	def setup_action_observation_space(self) -> None:
		# cell 0: number of products in the used storage, cell 1: number of products in circulation
		self.max_storage = 1e2
		self.observation_space = gym.spaces.Box(np.array([0, 0]), np.array([self.max_storage, 10 * self.max_storage]), dtype=np.float64)
		self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE)))

	def get_competitor_list(self) -> list:
		return []

	def reset_market_state(self) -> list:
		return [int(np.random.rand() * self.max_storage), int(5 * np.random.rand() * self.max_storage)]

	def action_to_array(self, action) -> np.array:
		# cell 0: price for second-hand-product, cell 1: price for new product (with rebuy price cell 3: rebuy price)
		return np.array(action) + 1

	def choose_customer(self) -> Customer:
		return customer.CustomerCircular()

	def choose_owner(self) -> Owner:
		return owner.OwnerReturn()

	def consider_owners_return(self, offer, profits) -> None:
		assert self.owner is not None, 'please choose an owner'
		for _ in range(int(0.05 * self.state[1])):
			owner_action = self.owner.consider_return()
			if owner_action == 0:
				# Owner throws away his object
				self.state[1] -= 1
			elif owner_action == 1:
				# Owner returns product to the agent

				# check if storage is full
				if self.state[0] < self.max_storage:
					self.state[0] += 1
				self.state[1] -= 1

	def complete_purchase(self, offers, profits, customer_buy) -> None:
		assert len(profits) == 1
		assert 0 < customer_buy and customer_buy <= 2
		if customer_buy == 1:
			if self.state[0] >= 1:
				# Increase the profit and decrease the storage
				profits[0] += offers[0]
				self.state[0] -= 1
			else:
				# Punish the agent for not having enough second-hand-products
				profits[0] -= 2 * ut.MAX_PRICE
		elif customer_buy == 2:
			profits[0] += offers[1] - ut.PRODUCTION_PRICE
			# One more product is in circulation now
			self.state[1] = min(self.state[1] + 1, 10 * self.max_storage)

	def consider_storage_costs(self, profits) -> None:
		profits[0] -= self.state[0] / 2  # Storage costs per timestep


class CircularEconomyRebuyPrice(CircularEconomy):
	def setup_action_observation_space(self) -> None:
		super().setup_action_observation_space()
		self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE)))

	def choose_owner(self) -> Owner:
		return owner.OwnerRebuy()

	def consider_owners_return(self, offer, profits) -> None:
		# just like with the customer the probabilities are set beforehand to improve performance
		assert self.owner is not None, 'please choose an owner'

		for _ in range(int(0.05 * self.state[1])):
			self.owner.set_probabilities_from_offer(offer)
			owner_action = self.owner.consider_return()
			if owner_action == 1:
				# Owner throws away his object
				self.state[1] -= 1
			elif owner_action == 2:
				# Owner returns product to the agent

				# check if storage is full
				if self.state[0] < self.max_storage:
					self.state[0] += 1
				self.state[1] -= 1
				profits[0] -= offer[2]
