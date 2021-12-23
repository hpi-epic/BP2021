#!/usr/bin/env python3
import copy
from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np

import customer
import owner
import utils_sim_market as ut
import vendors
from customer import Customer
from owner import Owner

# An offer is a market state that contains all prices and qualities

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
		assert (self.observation_space and self.action_space), 'Your observation or action space is not defined'
		# Make sure that variables such as state, customer are known
		self.reset()

	# The number of competitors plus the agent
	def get_number_of_vendors(self) -> int:
		"""This method returns the number of competitors plus the agent.

		Returns:
			int: number of competitors plus the agent
		"""
		return len(self.competitors) + 1

	def reset(self) -> np.array:
		"""This method is required by the gym library.
		It is called to reset the market before each episode.

		Returns:
			np.array: The initial observation of the market.
		"""
		self.step_counter = 0

		self.reset_common_state()

		self.vendor_specific_state = [self.reset_vendor_specific_state() for _ in range(self.get_number_of_vendors())]
		self.vendor_actions = [self.reset_vendor_actions() for _ in range(self.get_number_of_vendors())]

		self.customer = self.choose_customer()
		self.owner = self.choose_owner()

		return self.observation()

	def simulate_customers(self, profits, offers, number_of_customers) -> None:
		"""Here customers are simulated, the procducts offered by the vendors get sold to n customers.
		The profits for each vendor get saved to the profits array.

		Args:
			profits (np.array): The profits of the customers get saved to this array
			offers (np.array): this array contains the offers of the vendors. It has to be compatible with the customers used.
			number_of_customers (int): the number of customers eager to buy each step.
		"""
		probability_distribution = self.customer.generate_purchase_probabilities_from_offer(offers, self.get_offer_length_per_vendor())
		assert isinstance(probability_distribution, np.ndarray), 'Your method in customer must return a np.array!'
		assert len(probability_distribution) == 1 + (1 if isinstance(self, LinearEconomy) else 2) * self.get_number_of_vendors(), 'The probability distribution must have one entry for buy_nothing and one or two entries for every vendor. One entry if it is a linear economy (with only one price) or a circular economy with the option to buy refurbished or new.'

		for _ in range(number_of_customers):
			customer_decision = ut.shuffle_from_probabilities(probability_distribution)
			if customer_decision != 0:
				self.complete_purchase(profits, customer_decision - 1)
			else:
				self.output_dict['customer/buy_nothing'] += 1

	def step(self, action) -> Tuple[np.array, np.float64, bool, dict]:
		"""This method is called to simulate the market between actions by the agent.
		It is part of the gym library for reinforcement learning.
		It is pretty generic and configured by overwriting the abstract and empty methods.

		Args:
			action (np.array): The action of the agent. In discrete case: the action must be between 0 and number of actions -1.
			Note that you must add one to this price to get the real price!

		Returns:
			Tuple[np.array, np.float64, bool, dict]: A Tuple, containing the observation the agents makes right before his next action, the reward he made between these actions, a flag indicating if the market closes and information about the market for logging purposes.
		"""
		assert self.action_space.contains(action), f'{action} ({type(action)}) invalid'

		self.vendor_actions[0] = action

		self.step_counter += 1

		profits = [0] * self.get_number_of_vendors()

		self.output_dict = {'customer/buy_nothing': 0}
		self.initialize_output_dict()

		customers_per_vendor_iteration = int(np.floor(ut.NUMBER_OF_CUSTOMERS / self.get_number_of_vendors()))
		for i in range(self.get_number_of_vendors()):
			self.simulate_customers(profits, self.generate_customer_offer(), customers_per_vendor_iteration)
			if self.owner is not None:
				self.simulate_owners(profits, self.generate_customer_offer())

			# the competitor, which turn it is, will update its pricing
			if i < len(self.competitors):
				action_competitor_i = self.competitors[i].policy(self.observation(i + 1))
				assert self.action_space.contains(action_competitor_i), 'This vendor does not deliver a suitable action'
				self.vendor_actions[i + 1] = action_competitor_i

		self.consider_storage_costs(profits)

		self.ensure_output_dict_has('profits/all', profits)
		is_done = self.step_counter >= ut.EPISODE_LENGTH
		return self.observation(), profits[0], is_done, copy.deepcopy(self.output_dict)

	def observation(self, vendor_view=0) -> np.array:
		"""This method creates a different view of the market for every vendor.
		Each one sees every others vendors specific state, their actions and the global state.
		At the beginning of the array you have the common state.
		Afterwards you have the vendor specific state for the vendor with index vendor_view but NOT its actions from prior steps.
		Then, all other vendors follow with their actions and vendor specific state.

		Args:
			vendor_view (int, optional): Index of the vendor whose view we create. Defaults to 0.

		Returns:
			np.array: the view for the vendor with index vendor_view
		"""
		# observaton is the array containing the global state. We concatenate everything relevant to it, then return it.
		observation = self.get_common_state_array()
		assert isinstance(observation, np.ndarray), 'get_common_state_array must return a np-Array'

		# first the action and state of the of the vendor whose view we create will be added
		if self.vendor_specific_state[vendor_view] is not None:
			observation = np.concatenate((observation, np.array(self.vendor_specific_state[vendor_view], ndmin=1)), dtype=np.float64)

		# the rest of the vendors actions and states will be added
		for vendor_index in range(self.get_number_of_vendors()):
			if vendor_index == vendor_view:
				continue
			observation = np.concatenate((observation, np.array(self.vendor_actions[vendor_index], ndmin=1)), dtype=np.float64)
			if self.vendor_specific_state[vendor_index] is not None:
				observation = np.concatenate((observation, np.array(self.vendor_specific_state[vendor_index], ndmin=1)), dtype=np.float64)

		# The observation has to be part of the observation_space defined by the market
		assert self.observation_space.contains(observation), f'{observation} ({type(observation)}) invalid observation'
		return observation

	def generate_customer_offer(self) -> np.array:
		"""This methods maps the internal state to an array which is presented to the customers.
		It includes all information customers will use for their decisions.
		At the beginning of the array you have the common state.
		Afterwards you have the action and vendor specific state for all vendors.
		"""
		offer = self.get_common_state_array()
		assert isinstance(offer, np.ndarray), 'get_common_state_array must return a np-Array'
		for vendor_index in range(self.get_number_of_vendors()):
			offer = np.concatenate((offer, np.array(self.vendor_actions[vendor_index], ndmin=1)), dtype=np.float64)
			if self.vendor_specific_state[vendor_index] is not None:
				offer = np.concatenate((offer, np.array(self.vendor_specific_state[vendor_index], ndmin=1)), dtype=np.float64)
		return offer

	def reset_common_state(self) -> None:
		pass

	def get_common_state_array(self) -> np.array:
		return np.array([])

	def reset_vendor_specific_state(self) -> None:
		return None

	@abstractmethod
	def setup_action_observation_space(self) -> None:  # pragma: no cover
		raise NotImplementedError

	@abstractmethod
	def get_competitor_list(self) -> list:  # pragma: no cover
		raise NotImplementedError

	def consider_storage_costs(self, profits) -> None:
		pass

	def choose_owner(self):
		return None

	def get_offer_length_per_vendor(self) -> int:
		"""This methods generates the number of fields each vendor takes in the offer array

		Returns:
			int: The offer lenght is the sum of the number of fields required to encode the action and the length of the encoding of vendor specific state.
		"""
		action_encoding_length = 1 if isinstance(self.action_space, gym.spaces.Discrete) else len(self.action_space)
		if self.vendor_specific_state[0] is None:
			vendor_specific_state_encoding_length = 0
		else:
			vendor_specific_state_encoding_length = len(self.vendor_specific_state[0])
		return action_encoding_length + vendor_specific_state_encoding_length

	def ensure_output_dict_has(self, name, init_for_all_vendors=None) -> None:
		"""Ensures that the output_dict has an entry with the given name and creates an entry otherwise.
		If you pass a parameter for init_for_all_vendors, that will be interpreted as creating a dict with the passed array as content.

		Args:
			name (string): name of the dict entry which should be checked
			init_for_all_vendors (list, optional): initialization values for all vendors in this entry. Defaults to None.
		"""
		if init_for_all_vendors is not None:
			assert isinstance(init_for_all_vendors, list) and len(init_for_all_vendors) == self.get_number_of_vendors(), 'make sure you pass an array with length of number of vendors'
		if name not in self.output_dict:
			if init_for_all_vendors is None:
				self.output_dict[name] = 0
			else:
				self.output_dict[name] = dict(zip(['vendor_' + str(i) for i in range(self.get_number_of_vendors())], init_for_all_vendors))


class LinearEconomy(SimMarket, ABC):

	def setup_action_observation_space(self) -> None:
		"""The observation array has the following format:
		cell 0: quality of that vendor from whose perspective the observation is generated.
		following odd cells: price of an other vendor
		following even cells: quality of an other competitor

		The action space is discrete with as many actions as prices.
		"""
		self.observation_space = gym.spaces.Box(
			np.array([0.0] * (len(self.competitors) * 2 + 1)),
			np.array([ut.MAX_QUALITY] + [ut.MAX_PRICE, ut.MAX_QUALITY] * len(self.competitors)),
			dtype=np.float64)

		self.action_space = gym.spaces.Discrete(ut.MAX_PRICE)

	def reset_vendor_specific_state(self) -> list:
		return [ut.shuffle_quality()]

	def choose_customer(self) -> Customer:
		return customer.CustomerLinear()

	def reset_vendor_actions(self) -> int:
		"""Resets the price in the linear economy

		Returns:
			int: price of the new product
		"""
		return ut.PRODUCTION_PRICE + 1

	def complete_purchase(self, profits, chosen_vendor) -> None:
		"""The method handles the customer's decision by raising the profit by the price paid minus the produtcion price.

		Args:
			profits (np.array(int)): An array containing the profits of all vendors.
			chosen_vendor (int): Indicates the customer's decision.
		"""

		profits[chosen_vendor] += self.vendor_actions[chosen_vendor] - ut.PRODUCTION_PRICE
		self.output_dict['customer/purchases']['vendor_' + str(chosen_vendor)] += 1

	def initialize_output_dict(self):
		"""Initializes the entries for state and quality in the output dict for the linear economy
		"""
		self.ensure_output_dict_has('state/quality', [self.vendor_specific_state[i][0] for i in range(self.get_number_of_vendors())])

		self.ensure_output_dict_has('customer/purchases', [0] * self.get_number_of_vendors())


class ClassicScenario(LinearEconomy):

	def get_competitor_list(self) -> list:
		return [vendors.CompetitorLinearRatio1()]


class MultiCompetitorScenario(LinearEconomy):

	def get_competitor_list(self) -> list:
		return [
			vendors.CompetitorLinearRatio1(),
			vendors.CompetitorRandom(),
			vendors.CompetitorJust2Players(),
		]


class CircularEconomy(SimMarket):

	# currently monopoly
	def setup_action_observation_space(self) -> None:
		# cell 0: number of products in the used storage, cell 1: number of products in circulation
		self.max_storage = 1e2
		self.max_circulation = 10 * self.max_storage
		self.observation_space = gym.spaces.Box(np.array([0, 0] + [0, 0, 0] * len(self.competitors)), np.array([self.max_circulation, self.max_storage] + [ut.MAX_PRICE, ut.MAX_PRICE, self.max_storage] * len(self.competitors)), dtype=np.float64)
		self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE)))

	def reset_vendor_specific_state(self) -> list:
		"""In the circular economy, the vendor specific state contains the number of products in storage for each vendor

		Returns:
			list: a list with only one item which contains the number of elements in the storage of one specific vendor.
			It is chosen randomly between 0 and max_storage
		"""
		return [int(np.random.rand() * self.max_storage)]

	def reset_common_state(self) -> None:
		self.in_circulation = int(5 * np.random.rand() * self.max_storage)

	def get_common_state_array(self) -> np.array:
		return np.array([self.in_circulation])

	def reset_vendor_actions(self) -> tuple:
		"""Resets the prices in the circular economy (without rebuy price)

		Returns:
			tuple: (refurbished_price, new_price)
		"""
		return (ut.PRODUCTION_PRICE, ut.PRODUCTION_PRICE + 1)

	def choose_customer(self) -> Customer:
		return customer.CustomerCircular()

	def choose_owner(self) -> Owner:
		return owner.UniformDistributionOwner()

	def throw_away(self) -> None:
		"""The call of this method will decrease the in_circulation counter by one.
		Call it if one of your owners decided to throw away his product.
		"""
		self.output_dict['owner/throw_away'] += 1
		self.in_circulation -= 1

	def transfer_product_to_storage(self, vendor, profits=None, rebuy_price=0) -> None:
		"""Handles the transfer of a used product to the storage after it got bought by the vendor. It respects the storage capacity and adjusts the profit the vendor makes.

		Args:
			vendor (int): The index of the vendor that bought the product.
			profits (np.array(int), optional): The proftits of all vendors. Only the specific proftit of the given vendor is needed. Defaults to None.
			rebuy_price (int, optional): the price to which the used product is bought. Defaults to 0.
		"""
		self.output_dict['owner/rebuys']['vendor_' + str(vendor)] += 1

		self.vendor_specific_state[vendor][0] = min(self.vendor_specific_state[vendor][0] + 1, self.max_storage)  # receive the product only if you have space for it. Otherwise throw it away.
		self.in_circulation -= 1
		if profits is not None:
			self.output_dict['profits/rebuy_cost']['vendor_' + str(vendor)] -= rebuy_price
			profits[vendor] -= rebuy_price

	def simulate_owners(self, profits, offer) -> None:
		"""The process of owners selling their used products to the vendor. It is prepared for multiple vendor scenarios but is still part of a monopoly.

		Args:
			profits (np.array(int)): The profits of the vendor.
			offer (np.array): The offers of the vendor.
		"""
		assert self.owner is not None, 'please choose an owner'
		return_probabilities = self.owner.generate_return_probabilities_from_offer(offer, self.get_offer_length_per_vendor())
		assert isinstance(return_probabilities, np.ndarray) and len(return_probabilities) == 2 + self.get_number_of_vendors()

		number_of_owners = int(0.05 * self.in_circulation / self.get_number_of_vendors())
		for _ in range(number_of_owners):
			owner_action = ut.shuffle_from_probabilities(return_probabilities)

			# owner_action 0 means holding the product, so nothing happens
			if owner_action == 1:
				self.throw_away()
			elif owner_action >= 2:
				rebuy_price = self.get_rebuy_price(owner_action - 2)
				self.transfer_product_to_storage(owner_action - 2, profits, rebuy_price)

	def get_rebuy_price(self, _) -> int:
		return 0

	def complete_purchase(self, profits, customer_decision) -> None:
		"""The method handles the customer's decision by raising the profit by the price paid minus the produtcion price.
		It also handles the storage of used products.

		Args:
			profits (np.array(int)): The profits of all vendors.
			customer_decision (int): Indicates the customer's decision.
		"""
		assert 0 <= customer_decision and customer_decision < 2 * self.get_number_of_vendors(), 'Invalid action of the customer! Note that you have two options per vendor!'

		chosen_vendor = int(np.floor(customer_decision / 2))
		if customer_decision % 2 == 0:
			self.output_dict['customer/purchases_refurbished']['vendor_' + str(chosen_vendor)] += 1
			if self.vendor_specific_state[chosen_vendor][0] >= 1:
				# Increase the profit and decrease the storage
				profits[chosen_vendor] += self.vendor_actions[chosen_vendor][0]
				self.output_dict['profits/by_selling_refurbished']['vendor_' + str(chosen_vendor)] += self.vendor_actions[chosen_vendor][0]
				self.vendor_specific_state[chosen_vendor][0] -= 1
			else:
				# Punish the agent for not having enough second-hand-products
				profits[chosen_vendor] -= 2 * ut.MAX_PRICE
				self.output_dict['profits/by_selling_refurbished']['vendor_' + str(chosen_vendor)] -= 2 * ut.MAX_PRICE
		else:
			self.output_dict['customer/purchases_new']['vendor_' + str(chosen_vendor)] += 1
			profits[chosen_vendor] += self.vendor_actions[chosen_vendor][1] - ut.PRODUCTION_PRICE
			self.output_dict['profits/by_selling_new']['vendor_' + str(chosen_vendor)] += self.vendor_actions[chosen_vendor][1] - ut.PRODUCTION_PRICE
			# One more product is in circulation now, but only 10 times the amount of storage space we have
			self.in_circulation = min(self.in_circulation + 1, self.max_circulation)

	def consider_storage_costs(self, profits) -> None:
		"""The method handles the storage costs. they depend on the amount of refurbished products in storage.

		Args:
			profits (np.array(int)): The profits of all vendors.
		"""
		for vendor in range(self.get_number_of_vendors()):
			storage_cost_per_timestep = -self.vendor_specific_state[vendor][0] / 2
			profits[vendor] += storage_cost_per_timestep
			self.output_dict['profits/storage_cost']['vendor_' + str(vendor)] = storage_cost_per_timestep / 2

	def initialize_output_dict(self):
		"""Initialize the output_dict with the state of the environment and the actions the agents takes.
		Furthermore, the dictionary entries for all events which shall be monitored in the market are initialized.
		"""
		self.output_dict['state/in_circulation'] = self.in_circulation
		self.ensure_output_dict_has('state/in_storage', [self.vendor_specific_state[vendor][0] for vendor in range(self.get_number_of_vendors())])
		self.ensure_output_dict_has('actions/price_refurbished', [self.vendor_actions[vendor][0] for vendor in range(self.get_number_of_vendors())])
		self.ensure_output_dict_has('actions/price_new', [self.vendor_actions[vendor][1] for vendor in range(self.get_number_of_vendors())])

		self.ensure_output_dict_has('owner/throw_away')
		self.ensure_output_dict_has('owner/rebuys', [0] * self.get_number_of_vendors())
		self.ensure_output_dict_has('profits/rebuy_cost', [0] * self.get_number_of_vendors())

		self.ensure_output_dict_has('customer/purchases_refurbished', [0] * self.get_number_of_vendors())
		self.ensure_output_dict_has('customer/purchases_new', [0] * self.get_number_of_vendors())
		self.ensure_output_dict_has('profits/by_selling_refurbished', [0] * self.get_number_of_vendors())
		self.ensure_output_dict_has('profits/by_selling_new', [0] * self.get_number_of_vendors())

		self.ensure_output_dict_has('profits/storage_cost', [0] * self.get_number_of_vendors())


class CircularEconomyMonopolyScenario(CircularEconomy):
	def get_competitor_list(self) -> list:
		return []


class CircularEconomyRebuyPrice(CircularEconomy):

	def setup_action_observation_space(self) -> None:
		super().setup_action_observation_space()
		self.observation_space = gym.spaces.Box(np.array([0, 0] + [0, 0, 0, 0] * len(self.competitors)), np.array([self.max_circulation, self.max_storage] + [ut.MAX_PRICE, ut.MAX_PRICE, ut.MAX_PRICE, self.max_storage] * len(self.competitors)), dtype=np.float64)
		self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE), gym.spaces.Discrete(ut.MAX_PRICE)))

	def reset_vendor_actions(self) -> tuple:
		"""Resets the prices in the circular economy with rebuy prices.

		Returns:
			tuple: (refurbished_price, new_price, rebuy_price)
		"""
		return (ut.PRODUCTION_PRICE, ut.PRODUCTION_PRICE + 1, 1)

	def choose_owner(self) -> Owner:
		return owner.OwnerRebuy()

	def initialize_output_dict(self) -> None:
		"""Extends the the output_dict initialized by the of the superclass with entries concerning the rebuy price and cost.
		"""
		super().initialize_output_dict()
		self.ensure_output_dict_has('actions/price_rebuy', [self.vendor_actions[vendor][2] for vendor in range(self.get_number_of_vendors())])

		self.ensure_output_dict_has('profits/rebuy_cost', [0] * self.get_number_of_vendors())

	def get_rebuy_price(self, vendor_idx) -> int:
		return self.vendor_actions[vendor_idx][2]


class CircularEconomyRebuyPriceMonopolyScenario(CircularEconomyRebuyPrice):
	def get_competitor_list(self) -> list:
		return []


class CircularEconomyRebuyPriceOneCompetitor(CircularEconomyRebuyPrice):
	def get_competitor_list(self) -> list:
		return [vendors.RuleBasedCERebuyAgent()]
