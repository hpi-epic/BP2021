import gym
import numpy as np

import agents.vendors as vendors
import configuration.config as config
import configuration.utils as ut
import market.customer as customer
import market.owner as owner
from market.customer import Customer
from market.owner import Owner
from market.sim_market import SimMarket


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

	def _is_probability_distribution_fitting_exactly(self, probability_distribution) -> bool:
		"""
		The probability distribution must have one entry for buy_noting and two entries (purchases_new, purchases_refurbished) for every vendor.

		Args:
			probability_distribution (np.array):
				The probabilities that a customer either buys nothing or the refurbished or alternatively new product of a specific vendor.

		Returns:
			bool: Whether the probability_distribution fits into the CircularEconomy.
		"""
		return len(probability_distribution) == 1 + (2 * self._get_number_of_vendors())


class CircularEconomyMonopolyScenario(CircularEconomy):

	def _get_competitor_list(self) -> list:
		return []


class CircularEconomyRebuyPrice(CircularEconomy):

	def _setup_action_observation_space(self) -> None:
		super()._setup_action_observation_space()
		self.observation_space = gym.spaces.Box(
			np.array([0, 0] + [0, 0, 0, 0] * len(self.competitors)),
			np.array([self.max_circulation, self.max_storage] + [config.MAX_PRICE, config.MAX_PRICE,
				config.MAX_PRICE, self.max_storage] * len(self.competitors)),
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