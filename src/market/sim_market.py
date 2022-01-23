import copy
from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut
import market.customer as customer
import market.owner as owner
from market.customer import Customer
from market.owner import Owner

# An offer is a market state that contains all prices and qualities

# There are three kinds of state:
# First: a common state for all vendors
# Second: a state specific to one vendor
# Third: vendor's actions from the former round which needs to be saved and influence the other's decision e.g. prices


class SimMarket(gym.Env, ABC):
	"""
	The superclass to all market environments.

	Abstract class that cannot be instantiated.
	Inherits from `gym.env`.
	"""

	def __init__(self) -> None:
		"""
		Initialize a SimMarket instance.

		Set up needed values such as competitors and action/observation-space and reset the environment.
		"""
		self.competitors = self._get_competitor_list()
		# The agent's price does not belong to the observation_space any more because an agent should not depend on it
		self._setup_action_observation_space()
		self._owner = None
		self._customer = None
		# TODO: Better testing for the observation and action space
		assert (self.observation_space and self._action_space), 'Your observation or action space is not defined'
		# Make sure that variables such as state, customer are known
		self.reset()

	def _get_number_of_vendors(self) -> int:
		"""
		Return the number of competitors plus the agent.

		Returns:
			int: Number of competitors plus the agent.
		"""
		return len(self.competitors) + 1

	def reset(self) -> np.array:
		"""
		Reset the market before each episode.

		This method is required by the gym library.

		Returns:
			np.array: The initial observation of the market.
		"""
		self.step_counter = 0

		self._reset_common_state()

		self.vendor_specific_state = [self._reset_vendor_specific_state() for _ in range(self._get_number_of_vendors())]
		self.vendor_actions = [self._reset_vendor_actions() for _ in range(self._get_number_of_vendors())]

		self._customer = self._choose_customer()
		self._owner = self._choose_owner()

		return self._observation()

	def _simulate_customers(self, profits, offers, number_of_customers) -> None:
		"""
		Simulate the customers, the products offered by the vendors get sold to n customers.

		The profits for each vendor get saved to the profits array.

		Args:
			profits (np.array): The profits of the customers get saved to this array
			offers (np.array): this array contains the offers of the vendors. It has to be compatible with the customers used.
			number_of_customers (int): the number of customers eager to buy each step.
		"""
		probability_distribution = self._customer.generate_purchase_probabilities_from_offer(offers, self._get_offer_length_per_vendor())
		assert isinstance(probability_distribution, np.ndarray), 'generate_purchase_probabilities_from_offer must return an np.ndarray'
		assert len(probability_distribution) == 1 + (1 if isinstance(self, LinearEconomy) else 2) * self._get_number_of_vendors(), \
			"""The probability distribution must have one entry for buy_nothing and one or two entries for every vendor.
			One entry if it is a linear economy (with only one price) or a circular economy with the option to buy refurbished or new."""

		for _ in range(number_of_customers):
			customer_decision = ut.shuffle_from_probabilities(probability_distribution)
			if customer_decision != 0:
				self._complete_purchase(profits, customer_decision - 1)
			else:
				self.output_dict['customer/buy_nothing'] += 1

	def step(self, action) -> Tuple[np.array, np.float64, bool, dict]:
		"""
		Simulate the market between actions by the agent.

		It is part of the gym library for reinforcement learning.
		It is pretty generic and configured by overwriting the abstract and empty methods.

		Args:
			action (np.array): The action of the agent. In discrete case: the action must be between 0 and number of actions -1.
			Note that you must add one to this price to get the real price!

		Returns:
			Tuple[np.array, np.float64, bool, dict]: A Tuple,
			containing the observation the agents makes right before his next action,
			the reward he made between these actions,
			a flag indicating if the market closes and information about the market for logging purposes.
		"""
		assert self._action_space.contains(action), f'{action} ({type(action)}) invalid'

		self.vendor_actions[0] = action

		self.step_counter += 1

		profits = [0] * self._get_number_of_vendors()

		self.output_dict = {'customer/buy_nothing': 0}
		self._initialize_output_dict()

		customers_per_vendor_iteration = int(np.floor(config.NUMBER_OF_CUSTOMERS / self._get_number_of_vendors()))
		for i in range(self._get_number_of_vendors()):
			self._simulate_customers(profits, self._generate_customer_offer(), customers_per_vendor_iteration)
			if self._owner is not None:
				self._simulate_owners(profits, self._generate_customer_offer())

			# the competitor, which turn it is, will update its pricing
			if i < len(self.competitors):
				action_competitor_i = self.competitors[i].policy(self._observation(i + 1))
				assert self._action_space.contains(action_competitor_i), 'This vendor does not deliver a suitable action'
				self.vendor_actions[i + 1] = action_competitor_i

		self._consider_storage_costs(profits)

		self._ensure_output_dict_has('profits/all', profits)
		is_done = self.step_counter >= config.EPISODE_LENGTH
		return self._observation(), profits[0], is_done, copy.deepcopy(self.output_dict)

	def _observation(self, vendor_view=0) -> np.array:
		"""
		Create a different view of the market for every vendor.

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
		observation = self._get_common_state_array()
		assert isinstance(observation, np.ndarray), '_get_common_state_array must return an np.ndarray'

		# first the action and state of the of the vendor whose view we create will be added
		if self.vendor_specific_state[vendor_view] is not None:
			observation = np.concatenate((observation, np.array(self.vendor_specific_state[vendor_view], ndmin=1)), dtype=np.float64)

		# the rest of the vendors actions and states will be added
		for vendor_index in range(self._get_number_of_vendors()):
			if vendor_index == vendor_view:
				continue
			observation = np.concatenate((observation, np.array(self.vendor_actions[vendor_index], ndmin=1)), dtype=np.float64)
			if self.vendor_specific_state[vendor_index] is not None:
				observation = np.concatenate((observation, np.array(self.vendor_specific_state[vendor_index], ndmin=1)), dtype=np.float64)

		# The observation has to be part of the observation_space defined by the market
		assert self.observation_space.contains(observation), f'{observation} ({type(observation)}) invalid observation'
		return observation

	def _generate_customer_offer(self) -> np.array:
		"""
		Map the internal state to an array which is presented to the customers.

		It includes all information customers will use for their decisions.
		At the beginning of the array you have the common state.
		Afterwards you have the action and vendor specific state for all vendors.
		"""
		offer = self._get_common_state_array()
		assert isinstance(offer, np.ndarray), '_get_common_state_array must return an np.ndarray'
		for vendor_index in range(self._get_number_of_vendors()):
			offer = np.concatenate((offer, np.array(self.vendor_actions[vendor_index], ndmin=1)), dtype=np.float64)
			if self.vendor_specific_state[vendor_index] is not None:
				offer = np.concatenate((offer, np.array(self.vendor_specific_state[vendor_index], ndmin=1)), dtype=np.float64)
		return offer

	def _reset_common_state(self) -> None:
		pass

	def _get_common_state_array(self) -> np.array:
		return np.array([])

	@abstractmethod
	def _reset_vendor_specific_state(self) -> None:
		"""
		The implementation of this function varies between economy types.

		See also:
			`<market.sim_market.LinearEconomy._reset_vendor_specific_state>`
			`<market.sim_market.CircularEconomy._reset_vendor_specific_state>`
		"""
		raise NotImplementedError

	@abstractmethod
	def _reset_vendor_actions(self):
		"""
		Reset the price(s) in an economy.

		Returns:
			int or tuple: Price(s) of the new product.
		"""
		raise NotImplementedError

	@abstractmethod
	def _setup_action_observation_space(self) -> None:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def get_n_actions(self) -> int:  # pragma: no cover
		"""
		Return the number of actions agents should return in this marketplace.

		Depends on the `self.action_space`.

		Returns:
			int: The number of actions the agents should take in this marketplace.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def _get_competitor_list(self) -> list:  # pragma: no cover
		"""
		Get a list of all competitors in the current market scenario.

		TODO: This should get reworked since there no longer is a formal definition of 'competitor', since we see all vendors as agents.

		Returns:
			list: List containing instances of the competitors.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

	def _consider_storage_costs(self, profits) -> None:
		return None

	@abstractmethod
	def _choose_customer(self) -> None:
		"""
		Return the customer for this market scenario.

		Returns:
			Customer: An instance of a customer class from `<market.customer>`
		"""
		raise NotImplementedError

	def _choose_owner(self) -> None:
		"""
		Return the owner for this market scenario.

		Returns:
			Owner: An instance of an owner class from `<market.owner>`
			or
			None: If the market scenario does not support owners.
		"""
		return None

	@abstractmethod
	def _complete_purchase(self):
		"""The method handles the customer's decision by raising the profit by the price paid minus the produtcion price.

		Args:
			profits (np.array(int)): An array containing the profits of all vendors.
			chosen_vendor (int): Indicates the customer's decision.
		"""
		raise NotImplementedError

	@abstractmethod
	def _initialize_output_dict(self):
		"""
		Initialize the entries in the output dict for an economy.
		"""
		raise NotImplementedError

	def _get_offer_length_per_vendor(self) -> int:
		"""
		Generate the number of fields each vendor takes in the offers array.

		The offer length is the sum of the number of fields required to encode the action and the length of the encoding of vendor specific state.

		Returns:
			int: The offer length.
		"""
		action_encoding_length = 1 if isinstance(self._action_space, gym.spaces.Discrete) else len(self._action_space)
		if self.vendor_specific_state[0] is None:
			vendor_specific_state_encoding_length = 0
		else:
			vendor_specific_state_encoding_length = len(self.vendor_specific_state[0])
		return action_encoding_length + vendor_specific_state_encoding_length

	def _ensure_output_dict_has(self, name, init_for_all_vendors=None) -> None:
		"""
		Ensure that the output_dict has an entry with the given name and create an entry otherwise.

		If a parameter for init_for_all_vendors is passed, it will be interpreted as creating a dict with the passed array as content.

		Args:
			name (string): name of the dict entry which should be checked.
			init_for_all_vendors (list, optional): initialization values for all vendors in this entry. Defaults to None.
		"""
		if init_for_all_vendors is not None:
			assert isinstance(init_for_all_vendors, list) and len(init_for_all_vendors) == self._get_number_of_vendors(), \
				'make sure you pass a list with length of number of vendors'
		if name not in self.output_dict:
			if init_for_all_vendors is None:
				self.output_dict[name] = 0
			else:
				self.output_dict[name] = dict(zip(['vendor_' + str(i) for i in range(self._get_number_of_vendors())], init_for_all_vendors))


class LinearEconomy(SimMarket, ABC):

	def _setup_action_observation_space(self) -> None:
		"""
		The observation array has the following format:
		cell 0: quality of that vendor from whose perspective the observation is generated.
		following odd cells: price of an other vendor
		following even cells: quality of an other competitor

		The action space is discrete with as many actions as prices.
		"""
		self.observation_space = gym.spaces.Box(
			np.array([0.0] * (len(self.competitors) * 2 + 1)),
			np.array([config.MAX_QUALITY] + [config.MAX_PRICE, config.MAX_QUALITY] * len(self.competitors)),
			dtype=np.float64)

		self._action_space = gym.spaces.Discrete(config.MAX_PRICE)

	def _reset_vendor_specific_state(self) -> list:
		"""
		Return a list containing a randomized quality value of the product the vendor is selling.

		Returns:
			list: a list containing the quality value of the product.

		See also:
			`configuration.utils.shuffle_quality`
		"""
		return [ut.shuffle_quality()]

	def _choose_customer(self) -> Customer:
		return customer.CustomerLinear()

	def _reset_vendor_actions(self) -> int:
		"""
		Reset the price in the linear economy.

		Returns:
			int: The new price.
		"""
		return config.PRODUCTION_PRICE + 1

	def _complete_purchase(self, profits, chosen_vendor) -> None:
		profits[chosen_vendor] += self.vendor_actions[chosen_vendor] - config.PRODUCTION_PRICE
		self.output_dict['customer/purchases']['vendor_' + str(chosen_vendor)] += 1

	def _initialize_output_dict(self):
		self._ensure_output_dict_has('state/quality', [self.vendor_specific_state[i][0] for i in range(self._get_number_of_vendors())])

		self._ensure_output_dict_has('customer/purchases', [0] * self._get_number_of_vendors())

	def get_n_actions(self):
		return self._action_space.n


class ClassicScenario(LinearEconomy):

	def _get_competitor_list(self) -> list:
		return [vendors.CompetitorLinearRatio1()]


class MultiCompetitorScenario(LinearEconomy):

	def _get_competitor_list(self) -> list:
		return [
			vendors.CompetitorLinearRatio1(),
			vendors.CompetitorRandom(),
			vendors.CompetitorJust2Players(),
		]


class CircularEconomy(SimMarket):

	def _setup_action_observation_space(self) -> None:
		# cell 0: number of products in the used storage, cell 1: number of products in circulation
		self.max_storage = 1e2
		self.max_circulation = 10 * self.max_storage
		self.observation_space = gym.spaces.Box(
			np.array([0, 0] + [0, 0, 0] * len(self.competitors)),
			np.array([self.max_circulation, self.max_storage] + [config.MAX_PRICE, config.MAX_PRICE, self.max_storage] * len(self.competitors)),
			dtype=np.float64)
		self._action_space = gym.spaces.Tuple((gym.spaces.Discrete(config.MAX_PRICE), gym.spaces.Discrete(config.MAX_PRICE)))

	def _reset_vendor_specific_state(self) -> list:
		"""
		Return a list containing a randomized number of products in storage.

		Returns:
			list: a list with only the number of elements in the storage of one specific vendor.
			It is chosen randomly between 0 and `max_storage`.
		"""
		return [int(np.random.rand() * self.max_storage)]

	def _reset_common_state(self) -> None:
		self.in_circulation = int(5 * np.random.rand() * self.max_storage)

	def _get_common_state_array(self) -> np.array:
		return np.array([self.in_circulation])

	def _reset_vendor_actions(self) -> tuple:
		"""
		Reset the prices in the circular economy (without rebuy price)

		Returns:
			tuple: (refurbished_price, new_price)
		"""
		return (config.PRODUCTION_PRICE, config.PRODUCTION_PRICE + 1)

	def _choose_customer(self) -> Customer:
		return customer.CustomerCircular()

	def _choose_owner(self) -> Owner:
		return owner.UniformDistributionOwner()

	def _throw_away(self) -> None:
		"""
		The call of this method will decrease the in_circulation counter by one.
		Call it if one of your owners decided to throw away his product.
		"""
		self.output_dict['owner/throw_away'] += 1
		self.in_circulation -= 1

	def _transfer_product_to_storage(self, vendor, profits=None, rebuy_price=0) -> None:
		"""
		Handles the transfer of a used product to the storage after it got bought by the vendor.
		It respects the storage capacity and adjusts the profit the vendor makes.

		Args:
			vendor (int): The index of the vendor that bought the product.
			profits (np.array(int), optional): The proftits of all vendors.
			Only the specific proftit of the given vendor is needed. Defaults to None.
			rebuy_price (int, optional): the price to which the used product is bought. Defaults to 0.
		"""
		self.output_dict['owner/rebuys']['vendor_' + str(vendor)] += 1
		# receive the product only if you have space for it. Otherwise throw it away.
		self.vendor_specific_state[vendor][0] = min(self.vendor_specific_state[vendor][0] + 1, self.max_storage)
		self.in_circulation -= 1
		if profits is not None:
			self.output_dict['profits/rebuy_cost']['vendor_' + str(vendor)] -= rebuy_price
			profits[vendor] -= rebuy_price

	def _simulate_owners(self, profits, offer) -> None:
		"""
		The process of owners selling their used products to the vendor.
		It is prepared for multiple vendor scenarios but is still part of a monopoly.

		Args:
			profits (np.array(int)): The profits of the vendor.
			offer (np.array): The offers of the vendor.
		"""
		assert self._owner is not None, 'an owner must be set'
		return_probabilities = self._owner.generate_return_probabilities_from_offer(offer, self._get_offer_length_per_vendor())
		assert isinstance(return_probabilities, np.ndarray), 'return_probabilities must be an np.ndarray'
		assert len(return_probabilities) == 2 + self._get_number_of_vendors(), \
			'the length of return_probabilities must be the number of vendors plus 2'

		number_of_owners = int(0.05 * self.in_circulation / self._get_number_of_vendors())
		for _ in range(number_of_owners):
			owner_action = ut.shuffle_from_probabilities(return_probabilities)

			# owner_action 0 means holding the product, so nothing happens
			if owner_action == 1:
				self._throw_away()
			elif owner_action >= 2:
				rebuy_price = self._get_rebuy_price(owner_action - 2)
				self._transfer_product_to_storage(owner_action - 2, profits, rebuy_price)

	def _get_rebuy_price(self, _) -> int:
		return 0

	def _complete_purchase(self, profits, customer_decision) -> None:
		"""
		The method handles the customer's decision by raising the profit by the price paid minus the produtcion price.
		It also handles the storage of used products.

		Args:
			profits (np.array(int)): The profits of all vendors.
			customer_decision (int): Indicates the customer's decision.
		"""
		assert customer_decision >= 0 and customer_decision < 2 * self._get_number_of_vendors(), \
			'the customer_decision must be between 0 and 2 * the number of vendors, as each vendor offers a new and a refurbished product'

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
				profits[chosen_vendor] -= 2 * config.MAX_PRICE
				self.output_dict['profits/by_selling_refurbished']['vendor_' + str(chosen_vendor)] -= 2 * config.MAX_PRICE
		else:
			self.output_dict['customer/purchases_new']['vendor_' + str(chosen_vendor)] += 1
			profits[chosen_vendor] += self.vendor_actions[chosen_vendor][1] - config.PRODUCTION_PRICE
			self.output_dict['profits/by_selling_new']['vendor_' + str(chosen_vendor)] += (
				self.vendor_actions[chosen_vendor][1] - config.PRODUCTION_PRICE)
			# One more product is in circulation now, but only 10 times the amount of storage space we have
			self.in_circulation = min(self.in_circulation + 1, self.max_circulation)

	def _consider_storage_costs(self, profits) -> None:
		"""
		The method handles the storage costs. they depend on the amount of refurbished products in storage.

		Args:
			profits (np.array(int)): The profits of all vendors.
		"""
		for vendor in range(self._get_number_of_vendors()):
			storage_cost_per_timestep = -self.vendor_specific_state[vendor][0] / 2
			profits[vendor] += storage_cost_per_timestep
			self.output_dict['profits/storage_cost']['vendor_' + str(vendor)] = storage_cost_per_timestep / 2

	def _initialize_output_dict(self):
		"""
		Initialize the output_dict with the state of the environment and the actions the agents takes.

		Furthermore, the dictionary entries for all events which shall be monitored in the market are initialized.
		"""
		self.output_dict['state/in_circulation'] = self.in_circulation
		self._ensure_output_dict_has('state/in_storage',
			[self.vendor_specific_state[vendor][0] for vendor in range(self._get_number_of_vendors())])
		self._ensure_output_dict_has('actions/price_refurbished',
			[self.vendor_actions[vendor][0] for vendor in range(self._get_number_of_vendors())])
		self._ensure_output_dict_has('actions/price_new',
			[self.vendor_actions[vendor][1] for vendor in range(self._get_number_of_vendors())])

		self._ensure_output_dict_has('owner/throw_away')
		self._ensure_output_dict_has('owner/rebuys', [0] * self._get_number_of_vendors())
		self._ensure_output_dict_has('profits/rebuy_cost', [0] * self._get_number_of_vendors())

		self._ensure_output_dict_has('customer/purchases_refurbished', [0] * self._get_number_of_vendors())
		self._ensure_output_dict_has('customer/purchases_new', [0] * self._get_number_of_vendors())
		self._ensure_output_dict_has('profits/by_selling_refurbished', [0] * self._get_number_of_vendors())
		self._ensure_output_dict_has('profits/by_selling_new', [0] * self._get_number_of_vendors())

		self._ensure_output_dict_has('profits/storage_cost', [0] * self._get_number_of_vendors())

	def get_n_actions(self):
		n_actions = 1
		for id in range(len(self._action_space)):
			n_actions *= self._action_space[id].n
		return n_actions


class CircularEconomyMonopolyScenario(CircularEconomy):

	def _get_competitor_list(self) -> list:
		return []


class CircularEconomyRebuyPrice(CircularEconomy):

	def _setup_action_observation_space(self) -> None:
		super()._setup_action_observation_space()
		self.observation_space = gym.spaces.Box(
			np.array([0, 0] + [0, 0, 0, 0] * len(self.competitors)),
			np.array([self.max_circulation, self.max_storage] + [config.MAX_PRICE,
			config.MAX_PRICE,
			config.MAX_PRICE,
			self.max_storage] * len(self.competitors)),
			dtype=np.float64)
		self._action_space = gym.spaces.Tuple(
			(gym.spaces.Discrete(config.MAX_PRICE), gym.spaces.Discrete(config.MAX_PRICE), gym.spaces.Discrete(config.MAX_PRICE)))

	def _reset_vendor_actions(self) -> tuple:
		"""
		Resets the prices in the circular economy with rebuy prices.

		Returns:
			tuple: (refurbished_price, new_price, rebuy_price)
		"""
		return (config.PRODUCTION_PRICE, config.PRODUCTION_PRICE + 1, 1)

	def _choose_owner(self) -> Owner:
		return owner.OwnerRebuy()

	def _initialize_output_dict(self) -> None:
		"""
		Initialize the output_dict with the state of the environment and the actions the agents takes.

		Furthermore, the dictionary entries for all events which shall be monitored in the market are initialized.
		Also extend the the output_dict initialized by the superclass with entries concerning the rebuy price and cost.
		"""
		super()._initialize_output_dict()
		self._ensure_output_dict_has('actions/price_rebuy', [self.vendor_actions[vendor][2] for vendor in range(self._get_number_of_vendors())])

		self._ensure_output_dict_has('profits/rebuy_cost', [0] * self._get_number_of_vendors())

	def _get_rebuy_price(self, vendor_idx) -> int:
		return self.vendor_actions[vendor_idx][2]


class CircularEconomyRebuyPriceMonopolyScenario(CircularEconomyRebuyPrice):

	def _get_competitor_list(self) -> list:
		return []


class CircularEconomyRebuyPriceOneCompetitor(CircularEconomyRebuyPrice):

	def _get_competitor_list(self) -> list:
		return [vendors.RuleBasedCERebuyAgent()]
