import copy
from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np

import configuration.config as config
import configuration.utils as ut

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

	@abstractmethod
	def _is_probability_distribution_fitting_exactly(self, probability_distribution) -> None:
		"""
		The implementation of this function varies between economy types.

		See also:
			`<market.linear.linear_sim_market.LinearEconomy._is_probability_distribution_fitting_exactly`

			`<market.circular.circular_sim_market.CircularEconomy._is_probability_distribution_fitting_exactly>`
		"""
		raise NotImplementedError

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
		assert self._is_probability_distribution_fitting_exactly(probability_distribution)

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

	def get_actions_dimension(self) -> int:
		"""
		Get the dimension of the action space.
		This can be used to set the number of outputs for vendors with continuos action space.

		Returns:
			int: The dimension of the action space.
		"""
		if self._action_space.shape is not None:
			return 1
		else:
			return len(self._action_space)

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
