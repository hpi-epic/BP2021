from abc import ABC

import gym
import numpy as np

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_vendors as circular_vendors
import recommerce.market.owner as owner
from recommerce.configuration.common_rules import greater_zero_rule, non_negative_rule
from recommerce.market.circular.circular_customers import CustomerCircular
from recommerce.market.customer import Customer
from recommerce.market.owner import Owner
from recommerce.market.sim_market import SimMarket


class CircularEconomy(SimMarket, ABC):
	@staticmethod
	def get_competitor_classes() -> list:
		import recommerce.market.circular.circular_vendors as c_vendors
		return sorted(ut.filtered_class_str_from_dir('recommerce.market.circular.circular_vendors', dir(c_vendors), '.*CE.*Agent.*'))

	@staticmethod
	def get_configurable_fields() -> list:
		# TODO: reduce this list to only the required fields (remove max_quality)
		return [
			('max_storage', int, greater_zero_rule),
			('episode_length', int, greater_zero_rule),
			('max_price', int, greater_zero_rule),
			('max_quality', int, greater_zero_rule),
			('number_of_customers', int, (lambda number_of_customers: number_of_customers > 0 and number_of_customers % 2 == 0,
				'number_of_customers should be even and positive')),
			('production_price', int, non_negative_rule),
			('storage_cost_per_product', (int, float), non_negative_rule),
		]

	def _setup_action_observation_space(self, support_continuous_action_space: bool) -> None:
		# cell 0: number of products in the used storage, cell 1: number of products in circulation
		self.max_storage = self.config.max_storage
		self.max_circulation = 10 * self.max_storage
		self.observation_space = gym.spaces.Box(
			np.array([0, 0] + [0, 0, 0] * len(self.competitors), dtype=np.float32),
			np.array([self.max_circulation, self.max_storage] +
				[self.config.max_price, self.config.max_price, self.max_storage] * len(self.competitors), dtype=np.float32))

		if support_continuous_action_space:
			self.action_space = gym.spaces.Box(np.array([0] * 2, dtype=np.float32), np.array([self.config.max_price] * 2, dtype=np.float32))
		else:
			self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.config.max_price), gym.spaces.Discrete(self.config.max_price)))

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
		return (self.config.production_price, self.config.production_price + 1)

	def _choose_customer(self) -> Customer:
		return CustomerCircular()

	def _choose_owner(self) -> Owner:
		return owner.UniformDistributionOwner()

	def _throw_away(self, frequency) -> None:
		"""
		The call of this method will decrease the in_circulation counter by frequency-items.
		Call it with the amount of owners that decided to throw away their products.
		"""
		self._output_dict['owner/throw_away'] += frequency
		self.in_circulation -= frequency

	def _transfer_product_to_storage(self, vendor, profits, rebuy_price, frequency) -> None:
		"""
		Handles the transfer of a used product to the storage after it got bought by the vendor.
		It respects the storage capacity and adjusts the profit the vendor makes.
		Args:
			vendor (int): The index of the vendor that bought the product.
			profits (np.array(int)): The profits of all vendors. Only the specific profit of the given vendor is needed.
			rebuy_price (int): The price to which the used product is bought.
			frequency (int): The number of transferred items.
		"""
		assert profits is not None, 'profits must not be None'
		self._output_dict['owner/rebuys'][f'vendor_{vendor}'] += frequency
		# receive the product only if you have space for it. Otherwise throw it away. But you have to pay anyway.
		self.vendor_specific_state[vendor][0] = min(self.vendor_specific_state[vendor][0] + frequency, self.max_storage)
		self.in_circulation -= frequency
		rebuy_cost = frequency * rebuy_price
		self._output_dict['profits/rebuy_cost'][f'vendor_{vendor}'] -= rebuy_cost
		profits[vendor] -= rebuy_cost

	def _simulate_owners(self, profits) -> None:
		"""
		The process of owners selling their used products to the vendor.
		It is prepared for multiple vendor scenarios but is still part of a monopoly.
		Args:
			profits (np.array(int)): The profits of the vendor.
		"""
		assert self._owner is not None, 'an owner must be set'
		return_probabilities = self._owner.generate_return_probabilities_from_offer(
			self._get_common_state_array(), self.vendor_specific_state, self.vendor_actions)
		assert isinstance(return_probabilities, np.ndarray), 'return_probabilities must be an np.ndarray'
		assert len(return_probabilities) == 2 + self._number_of_vendors, \
			'the length of return_probabilities must be the number of vendors plus 2'

		number_of_owners = int(0.05 * self.in_circulation / self._number_of_vendors)
		owner_decisions = np.random.multinomial(number_of_owners, return_probabilities).tolist()

		# owner decisions can be as follows:
		# 0: Hold/Do nothing
		# 1: Throw away
		# x: Sell back to vendor x-2
		# for each action, `owner_decisions` has a frequency of how many customers chose that action
		self._throw_away(owner_decisions[1])

		for rebuy_vendor, frequency in enumerate(owner_decisions[2:]):
			rebuy_price = self._get_rebuy_price(rebuy_vendor)
			self._transfer_product_to_storage(rebuy_vendor, profits, rebuy_price, frequency)

	def _get_rebuy_price(self, _) -> int:
		return 0

	def _complete_purchase(self, profits, customer_decision, frequency) -> None:
		"""
		The method handles the customer's decision by raising the profit by the price paid minus the produtcion price.
		It also handles the storage of used products.
		Args:
			profits (np.array(int)): The profits of all vendors.
			customer_decision (int): Indicates the customer's decision.
			frequency (int): The number of items bought at this vendor.
		"""
		assert customer_decision >= 0 and customer_decision < 2 * self._number_of_vendors, \
			'the customer_decision must be between 0 and 2 * the number of vendors, as each vendor offers a new and a refurbished product'

		chosen_vendor = customer_decision // 2
		if customer_decision % 2 == 0:
			# Calculate how many refurbished can be sold
			possible_refurbished_sales = min(frequency, self.vendor_specific_state[chosen_vendor][0])

			# Increase the profit and decrease the storage
			profit = possible_refurbished_sales * self.vendor_actions[chosen_vendor][0]
			self.vendor_specific_state[chosen_vendor][0] -= possible_refurbished_sales
			profits[chosen_vendor] += profit
			self._output_dict['customer/purchases_refurbished'][f'vendor_{chosen_vendor}'] += possible_refurbished_sales
			self._output_dict['profits/by_selling_refurbished'][f'vendor_{chosen_vendor}'] += profit

			# Punish the agent for not having enough second-hand-products
			impossible_refurbished_sales = frequency - possible_refurbished_sales
			punishment = 2 * self.config.max_price * impossible_refurbished_sales
			profits[chosen_vendor] -= punishment
			self._output_dict['profits/by_selling_refurbished'][f'vendor_{chosen_vendor}'] -= punishment

		else:
			self._output_dict['customer/purchases_new'][f'vendor_{chosen_vendor}'] += frequency
			profit = frequency * (self.vendor_actions[chosen_vendor][1] - self.config.production_price)
			profits[chosen_vendor] += profit
			self._output_dict['profits/by_selling_new'][f'vendor_{chosen_vendor}'] += profit
			# The number of items in circulation is bounded
			self.in_circulation = min(self.in_circulation + frequency, self.max_circulation)

		assert self.vendor_specific_state[chosen_vendor][0] >= 0, 'Your code must ensure a non-negative storage'

	def _consider_storage_costs(self, profits) -> None:
		"""
		The method handles the storage costs. they depend on the amount of refurbished products in storage.
		Args:
			profits (np.array(int)): The profits of all vendors.
		"""
		for vendor in range(self._number_of_vendors):
			storage_cost_per_timestep = -self.vendor_specific_state[vendor][0] * self.config.storage_cost_per_product
			profits[vendor] += storage_cost_per_timestep
			self._output_dict['profits/storage_cost'][f'vendor_{vendor}'] = storage_cost_per_timestep

	def _initialize_output_dict(self):
		"""
		Initialize the _output_dict with the state of the environment and the actions the agents takes.
		Furthermore, the dictionary entries for all events which shall be monitored in the market are initialized.
		"""
		self._output_dict['state/in_circulation'] = self.in_circulation
		self._ensure_output_dict_has('state/in_storage',
			[self.vendor_specific_state[vendor][0] for vendor in range(self._number_of_vendors)])
		self._ensure_output_dict_has('actions/price_refurbished',
			[self.vendor_actions[vendor][0] for vendor in range(self._number_of_vendors)])
		self._ensure_output_dict_has('actions/price_new',
			[self.vendor_actions[vendor][1] for vendor in range(self._number_of_vendors)])

		self._ensure_output_dict_has('owner/throw_away')
		self._ensure_output_dict_has('owner/rebuys', [0] * self._number_of_vendors)
		self._ensure_output_dict_has('profits/rebuy_cost', [0] * self._number_of_vendors)

		self._ensure_output_dict_has('customer/purchases_refurbished', [0] * self._number_of_vendors)
		self._ensure_output_dict_has('customer/purchases_new', [0] * self._number_of_vendors)
		self._ensure_output_dict_has('profits/by_selling_refurbished', [0] * self._number_of_vendors)
		self._ensure_output_dict_has('profits/by_selling_new', [0] * self._number_of_vendors)

		self._ensure_output_dict_has('profits/storage_cost', [0] * self._number_of_vendors)

	def get_n_actions(self):
		n_actions = 1
		for id in range(len(self.action_space)):
			n_actions *= self.action_space[id].n
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
		return len(probability_distribution) == 1 + (2 * self._number_of_vendors)


class CircularEconomyMonopoly(CircularEconomy):
	"""
	This is a circular economy with only one vendor having the monopoly.
	"""
	@staticmethod
	def get_num_competitors() -> int:
		return 0

	def _get_competitor_list(self) -> list:
		return []


class CircularEconomyDuopoly(CircularEconomy):
	"""
	This is a circular economy with two vendors (one agent and one competitor) playing against each other.
	"""
	@staticmethod
	def get_num_competitors() -> int:
		return 1

	def _get_competitor_list(self) -> list:
		return [circular_vendors.RuleBasedCEAgent(config_market=self.config)]


class CircularEconomyOligopoly(CircularEconomy):
	"""
	This is a circular economy with multiple vendors.
	"""
	@staticmethod
	def get_num_competitors() -> int:
		return np.inf

	def _get_competitor_list(self) -> list:
		return [
			circular_vendors.RuleBasedCEAgent(config_market=self.config),
			circular_vendors.RuleBasedCEAgent(config_market=self.config),
			circular_vendors.FixedPriceCEAgent(config_market=self.config, fixed_price=(3, 5)),
			circular_vendors.FixedPriceCEAgent(config_market=self.config, fixed_price=(2, 6))
			]


class CircularEconomyRebuyPrice(CircularEconomy, ABC):
	@staticmethod
	def get_competitor_classes() -> list:
		import recommerce.market.circular.circular_vendors as c_vendors
		return sorted(ut.filtered_class_str_from_dir('recommerce.market.circular.circular_vendors', dir(c_vendors), '.*CERebuy.*Agent.*'))

	def _setup_action_observation_space(self, support_continuous_action_space: bool) -> None:
		super()._setup_action_observation_space(support_continuous_action_space)
		self.observation_space = gym.spaces.Box(
			np.array([0, 0] + [0, 0, 0, 0] * len(self.competitors), dtype=np.float32),
			np.array([self.max_circulation, self.max_storage] + [self.config.max_price, self.config.max_price,
				self.config.max_price, self.max_storage] * len(self.competitors), dtype=np.float32))

		if support_continuous_action_space:
			self.action_space = gym.spaces.Box(np.array([0] * 3, dtype=np.float32), np.array([self.config.max_price] * 3, dtype=np.float32))
		else:
			self.action_space = gym.spaces.Tuple(
				(gym.spaces.Discrete(self.config.max_price), gym.spaces.Discrete(self.config.max_price), gym.spaces.Discrete(self.config.max_price)))

	def _reset_vendor_actions(self) -> tuple:
		"""
		Resets the prices in the circular economy with rebuy prices.
		Returns:
			tuple: (refurbished_price, new_price, rebuy_price)
		"""
		return (self.config.production_price, self.config.production_price + 1, 1)

	def _choose_owner(self) -> Owner:
		return owner.OwnerRebuy()

	def _initialize_output_dict(self) -> None:
		"""
		Initialize the _output_dict with the state of the environment and the actions the agents takes.
		Furthermore, the dictionary entries for all events which shall be monitored in the market are initialized.
		Also extend the the _output_dict initialized by the superclass with entries concerning the rebuy price and cost.
		"""
		super()._initialize_output_dict()
		self._ensure_output_dict_has('actions/price_rebuy', [self.vendor_actions[vendor][2] for vendor in range(self._number_of_vendors)])

		self._ensure_output_dict_has('profits/rebuy_cost', [0] * self._number_of_vendors)

	def _get_rebuy_price(self, vendor_idx) -> int:
		return self.vendor_actions[vendor_idx][2]


class CircularEconomyRebuyPriceMonopoly(CircularEconomyRebuyPrice):
	"""
	This is a circular economy with rebuy price, so the vendor buys back its products from the customers.

	There is only one vendor.
	"""
	@staticmethod
	def get_num_competitors() -> int:
		return 0

	def _get_competitor_list(self) -> list:
		return []


class CircularEconomyRebuyPriceDuopoly(CircularEconomyRebuyPrice):
	"""
	This is a circular economy with rebuy price, so the vendors buy back their products from the customers.
	There are two vendors.
	"""
	@staticmethod
	def get_num_competitors() -> int:
		return 1

	def _get_competitor_list(self) -> list:
		return [circular_vendors.RuleBasedCERebuyAgentCompetitive(config_market=self.config)]


class CircularEconomyRebuyPriceOligopoly(CircularEconomyRebuyPrice):
	"""
	This is a circular economy with rebuy price, so the vendors buy back their products from the customers.
	There are multiple vendors.
	"""
	@staticmethod
	def get_num_competitors() -> int:
		return np.inf

	def _get_competitor_list(self) -> list:
		return [
			circular_vendors.RuleBasedCERebuyAgentCompetitive(config_market=self.config),
			circular_vendors.RuleBasedCERebuyAgent(config_market=self.config),
			circular_vendors.FixedPriceCERebuyAgent(config_market=self.config, fixed_price=(3, 6, 2)),
			circular_vendors.RuleBasedCERebuyAgentStorageMinimizer(config_market=self.config)
			]
