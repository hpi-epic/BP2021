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

		self.reset_common_state()

		self.vendor_specific_state = []
		for _ in range(self.n_vendors()):
			self.vendor_specific_state.append(self.reset_specific_vendor_state())

		self.vendors_actions = []
		for _ in range(self.n_vendors()):
			self.vendors_actions.append(self.reset_vendors_actions())

		self.customer = self.choose_customer()
		self.owner = self.choose_owner()

		return copy.deepcopy(self.observation())

	def simulate_customers(self, profits, offers, n) -> None:
		self.customer.set_probabilities_from_offers(offers)
		for _ in range(n):
			customer_buy = self.customer.buy_object(offers)
			if customer_buy != 0:
				self.complete_purchase(offers, profits, customer_buy)
			else:
				self.output_dict['customer/buy_nothing'] += 1

	def step(self, action) -> Tuple[np.array, np.float64, bool, dict]:
		# The action is the new price of the agent

		err_msg = '%r (%s) invalid' % (action, type(action))
		assert self.action_space.contains(action), err_msg
		self.vendors_actions[0] = action

		self.step_counter += 1

		profits = [0] * self.n_vendors()

		self.output_dict = {'customer/buy_nothing': 0}

		customers_per_vendor_iteration = int(np.floor(ut.NUMBER_OF_CUSTOMERS / self.n_vendors()))
		for i in range(self.n_vendors()):
			self.simulate_customers(profits, self.generate_customer_offer(), customers_per_vendor_iteration)
			self.simulate_owners(profits, self.generate_customer_offer())

			# the competitor, which turn it is, will update its pricing
			if i < len(self.competitors):
				action_competitor_i = self.competitors[i].policy(
					self.observation(i + 1)
				)
				assert self.action_space.contains(action_competitor_i), 'This vendor does not deliver a suitable action'
				self.vendors_actions[i + 1] = action_competitor_i

		self.consider_storage_costs(profits)

		self.ensure_output_dict_has('profits/all', profits)
		self.extend_dict_from_state()
		is_done = self.step_counter >= ut.EPISODE_LENGTH
		return copy.deepcopy(self.observation()), profits[0], is_done, self.output_dict

	def observation(self, vendor_view=0):
		obs = self.get_common_state_array()
		assert isinstance(obs, np.ndarray), 'get_common_state_array must return a np-Array'
		if self.vendor_specific_state[vendor_view] is not None:
			obs = np.concatenate((obs, np.array(self.vendor_specific_state[vendor_view], ndmin=1)), dtype=np.float64)
		for i in range(self.n_vendors()):
			if i == vendor_view:
				continue
			obs = np.concatenate((obs, np.array(self.vendors_actions[i], ndmin=1)), dtype=np.float64)
			if self.vendor_specific_state[i] is not None:
				obs = np.concatenate((obs, np.array(self.vendor_specific_state[i], ndmin=1)), dtype=np.float64)
		err_msg = '%r (%s) invalid observation' % (obs, type(obs))
		assert self.observation_space.contains(obs), err_msg
		return obs

	def generate_customer_offer(self):
		offer = self.get_common_state_array()
		assert isinstance(offer, np.ndarray), 'get_common_state_array must return a np-Array'
		for i in range(self.n_vendors()):
			offer = np.concatenate((offer, np.array(self.vendors_actions[i], ndmin=1)), dtype=np.float64)
			if self.vendor_specific_state[i] is not None:
				offer = np.concatenate((offer, np.array(self.vendor_specific_state[i], ndmin=1)), dtype=np.float64)
		return offer

	def reset_common_state(self) -> None:
		pass

	def get_common_state_array(self) -> np.array:
		return np.array([])

	def reset_specific_vendor_state(self) -> None:
		None

	def simulate_owners(self, *_) -> None:
		pass

	@abstractmethod
	def get_competitor_list(self) -> list:
		raise NotImplementedError

	def consider_storage_costs(self, profits) -> None:
		pass

	def choose_owner(self):
		pass

	def ensure_output_dict_has(self, name, init_for_all_vendors=None):
		if name not in self.output_dict:
			if init_for_all_vendors is None:
				self.output_dict[name] = 0
			else:
				self.output_dict[name] = dict(zip(['vendor_' + str(i) for i in range(self.n_vendors())], init_for_all_vendors))


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

	def reset_specific_vendor_state(self) -> list:
		return [ut.shuffle_quality()]

	def choose_customer(self) -> Customer:
		return customer.CustomerLinear()

	def reset_vendors_actions(self) -> list:
		return ut.PRODUCTION_PRICE + 1

	def complete_purchase(self, offers, profits, customer_buy) -> None:
		self.ensure_output_dict_has('customer/purchases', [0] * self.n_vendors())

		profits[customer_buy - 1] += (offers[(customer_buy - 1) * 2] - ut.PRODUCTION_PRICE)
		self.output_dict['customer/purchases']['vendor_' + str(customer_buy - 1)] += 1

	def extend_dict_from_state(self):
		self.ensure_output_dict_has('state/quality', [self.vendor_specific_state[i][0] for i in range(self.n_vendors())])


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

	def reset_common_state(self) -> None:
		self.in_storage = int(np.random.rand() * self.max_storage)
		self.in_circulation = int(5 * np.random.rand() * self.max_storage)

	def get_common_state_array(self) -> np.array:
		return np.array([self.in_storage, self.in_circulation])

	def reset_vendors_actions(self) -> int:
		return (ut.PRODUCTION_PRICE, ut.PRODUCTION_PRICE + 1)

	def choose_customer(self) -> Customer:
		return customer.CustomerCircular()

	def choose_owner(self) -> Owner:
		return owner.OwnerReturn()

	def throw_away(self) -> None:
		self.output_dict['owner/throw_away'] += 1
		self.in_circulation -= 1

	def transfer_product_to_storage(self, vendor, profits=None, rebuy_price=0) -> None:
		self.output_dict['owner/rebuys']['vendor_' + str(vendor)] += 1

		if self.in_storage < self.max_storage:
			self.in_storage += 1
		self.in_circulation -= 1
		if profits is not None:
			self.output_dict['profits/rebuy_cost']['vendor_' + str(vendor)] -= rebuy_price
			profits[vendor] -= rebuy_price

	def simulate_owners(self, *_) -> None:
		self.ensure_output_dict_has('owner/throw_away')
		self.ensure_output_dict_has('owner/rebuys', [0] * self.n_vendors())
		self.ensure_output_dict_has('profits/rebuy_cost', [0] * self.n_vendors())
		assert self.owner is not None, 'please choose an owner'
		for _ in range(int(0.05 * self.in_circulation / self.n_vendors())):
			owner_action = self.owner.consider_return()
			if owner_action == 0:
				self.throw_away()
			else:
				self.transfer_product_to_storage(owner_action - 1)

	def complete_purchase(self, offers, profits, customer_buy) -> None:
		assert len(profits) == 1
		assert 0 < customer_buy and customer_buy <= 2
		self.ensure_output_dict_has('customer/purchases_refurbished', [0] * self.n_vendors())
		self.ensure_output_dict_has('customer/purchases_new', [0] * self.n_vendors())
		self.ensure_output_dict_has('profits/by_selling_refurbished', [0] * self.n_vendors())
		self.ensure_output_dict_has('profits/by_selling_new', [0] * self.n_vendors())

		if customer_buy == 1:
			self.output_dict['customer/purchases_refurbished']['vendor_0'] += 1
			if self.in_storage >= 1:
				# Increase the profit and decrease the storage
				profits[0] += self.vendors_actions[0][0]
				self.output_dict['profits/by_selling_refurbished']['vendor_0'] += self.vendors_actions[0][0]
				self.in_storage -= 1
			else:
				# Punish the agent for not having enough second-hand-products
				profits[0] -= 2 * ut.MAX_PRICE
				self.output_dict['profits/by_selling_refurbished']['vendor_0'] -= 2 * ut.MAX_PRICE
		elif customer_buy == 2:
			self.output_dict['customer/purchases_new']['vendor_0'] += 1
			profits[0] += self.vendors_actions[0][1] - ut.PRODUCTION_PRICE
			self.output_dict['profits/by_selling_new']['vendor_0'] += self.vendors_actions[0][1] - ut.PRODUCTION_PRICE
			# One more product is in circulation now, but only 10 times the amount of storage space we have
			self.in_circulation = min(self.in_circulation + 1, 10 * self.max_storage)

	def consider_storage_costs(self, profits) -> None:
		assert self.n_vendors() == 1, 'This feature does not support more than one customer yet'
		self.ensure_output_dict_has('profits/storage_cost', [0] * self.n_vendors())
		profits[0] -= self.in_storage / 2  # Storage costs per timestep
		self.output_dict['profits/storage_cost']['vendor_0'] = -self.in_storage / 2

	def extend_dict_from_state(self):
		assert self.n_vendors() == 1, 'This feature does not support more than one customer'
		self.output_dict['state/in_circulation'] = self.in_circulation
		self.ensure_output_dict_has('state/in_storage', [self.in_storage])  # self.vendor_specific_state)
		self.ensure_output_dict_has('actions/price_refurbished', [self.vendors_actions[i][0] for i in range(self.n_vendors())])
		self.ensure_output_dict_has('actions/price_new', [self.vendors_actions[i][1] for i in range(self.n_vendors())])


class CircularEconomyRebuyPrice(CircularEconomy):
	def setup_action_observation_space(self) -> None:
		super().setup_action_observation_space()
		self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE)))

	def reset_vendors_actions(self) -> int:
		return (ut.PRODUCTION_PRICE, ut.PRODUCTION_PRICE + 1, 1)

	def choose_owner(self) -> Owner:
		return owner.OwnerRebuy()

	def simulate_owners(self, profits, offer) -> None:
		# just like with the customer the probabilities are set beforehand to improve performance
		assert self.owner is not None, 'please choose an owner'
		self.ensure_output_dict_has('owner/throw_away')
		self.ensure_output_dict_has('owner/rebuys', [0] * self.n_vendors())
		self.ensure_output_dict_has('profits/rebuy_cost', [0] * self.n_vendors())

		for _ in range(int(0.05 * self.in_circulation)):
			self.owner.set_probabilities_from_offer(offer)
			owner_action = self.owner.consider_return()
			if owner_action == 1:
				self.throw_away()
			elif owner_action >= 2:
				rebuy_price = self.vendors_actions[owner_action - 2][2]
				self.transfer_product_to_storage(owner_action - 2, profits, rebuy_price)

	def extend_dict_from_state(self):
		super().extend_dict_from_state()
		self.ensure_output_dict_has('actions/price_rebuy', [self.vendors_actions[i][2] for i in range(self.n_vendors())])
