from abc import ABC

import agents.vendors as vendors
import configuration.utils as ut
import gym
import market.customer as customer
import numpy as np
from configuration.hyperparameter_config import config
from market.customer import Customer
from market.sim_market import SimMarket


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
			np.array([config.max_quality] + [config.max_price, config.max_quality] * len(self.competitors)),
			dtype=np.float64)

		self._action_space = gym.spaces.Discrete(config.max_price)

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
		return config.production_price + 1

	def _complete_purchase(self, profits, chosen_vendor, frequency) -> None:
		profits[chosen_vendor] += frequency * (self.vendor_actions[chosen_vendor] - config.production_price)
		self._output_dict['customer/purchases'][f'vendor_{chosen_vendor}'] += frequency

	def _initialize_output_dict(self):
		self._ensure_output_dict_has('state/quality', [self.vendor_specific_state[i][0] for i in range(self._number_of_vendors)])

		self._ensure_output_dict_has('customer/purchases', [0] * self._number_of_vendors)

	def get_n_actions(self):
		return self._action_space.n

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
