from abc import ABC

import gym
import numpy as np

import recommerce.configuration.utils as ut
from recommerce.market.customer import Customer
from recommerce.market.linear.linear_customers import CustomerLinear
from recommerce.market.linear.linear_vendors import Just2PlayersLEAgent, LERandomAgent, LinearRatio1LEAgent
from recommerce.market.sim_market import SimMarket


class LinearEconomy(SimMarket, ABC):
	@staticmethod
	def get_competitor_classes() -> list:
		import recommerce.market.linear.linear_vendors as l_vendors
		return sorted(ut.filtered_class_str_from_dir('recommerce.market.linear.linear_vendors', dir(l_vendors), '.*LE.*Agent.*'))

	def _setup_action_observation_space(self, support_continuous_action_space: bool) -> None:
		"""
		The observation array has the following format:
		cell 0: quality of that vendor from whose perspective the observation is generated.
		following odd cells: price of an other vendor
		following even cells: quality of an other competitor
		The action space is discrete with as many actions as prices.

				Args:
			support_continuous_action_space (bool): If True, the action space will be continuous.
		"""
		self.observation_space = gym.spaces.Box(
			np.array([0.0] * (len(self.competitors) * 2 + 1), dtype=np.float32),
			np.array([self.config.max_quality] + [self.config.max_price, self.config.max_quality] * len(self.competitors), dtype=np.float32))

		if support_continuous_action_space:
			self.action_space = gym.spaces.Box(np.float32(0), np.float32(self.config.max_price))
		else:
			self.action_space = gym.spaces.Discrete(self.config.max_price)

	def _reset_vendor_specific_state(self) -> list:
		"""
		Return a list containing a randomized quality value of the product the vendor is selling.
		Returns:
			list: a list containing the quality value of the product.
		See also:
			`configuration.utils.shuffle_quality`
		"""
		return [ut.shuffle_quality(self.config)]

	def _choose_customer(self) -> Customer:
		return CustomerLinear()

	def _reset_vendor_actions(self) -> int:
		"""
		Reset the price in the linear economy.
		Returns:
			int: The new price.
		"""
		return self.config.production_price + 1

	def _complete_purchase(self, profits, chosen_vendor, frequency) -> None:
		profits[chosen_vendor] += frequency * (self.vendor_actions[chosen_vendor] - self.config.production_price)
		self._output_dict['customer/purchases'][f'vendor_{chosen_vendor}'] += frequency

	def _initialize_output_dict(self):
		self._ensure_output_dict_has('state/quality', [self.vendor_specific_state[i][0] for i in range(self._number_of_vendors)])

		self._ensure_output_dict_has('customer/purchases', [0] * self._number_of_vendors)

	def get_n_actions(self):
		return self.action_space.n

	def _is_probability_distribution_fitting_exactly(self, probability_distribution) -> bool:
		"""
		The probability distribution must have one entry for buy_nothing and one entry (purchases_new) for every vendor.
		Args:
			probability_distribution (np.array): The probabilities that a customer either buys nothing or the new product of a specific vendor.
		Returns:
			bool: Whether the probability_distribution fits into the LinearEcononmy.
		"""
		return len(probability_distribution) == 1 + self._number_of_vendors

	def _get_common_state_array(self) -> np.array:
		return np.array([])


class LinearEconomyDuopoly(LinearEconomy):
	"""
	This is a linear economy, with two vendors.
	"""
	@staticmethod
	def get_num_competitors() -> list:
		return 1

	def _get_competitor_list(self) -> list:
		return [LinearRatio1LEAgent(config=self.config)]


class LinearEconomyOligopoly(LinearEconomy):
	"""
	This is a linear economy, with multiple vendors.
	"""
	def _get_competitor_list(self) -> list:
		return [
			LinearRatio1LEAgent(config=self.config),
			LERandomAgent(config=self.config),
			Just2PlayersLEAgent(config=self.config),
		]
