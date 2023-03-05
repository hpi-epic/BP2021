from abc import ABC

import gym
import numpy as np
from numpy import ndarray

import recommerce.configuration.utils as ut
from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_even_rule, greater_zero_rule, \
    non_negative_rule
from recommerce.market.customer import Customer
from recommerce.market.linear.linear_customers import CustomerLinear
from recommerce.market.linear.linear_vendors import Just2PlayersLEAgent, LERandomAgent, LinearRatio1LEAgent
from recommerce.market.sim_market import SimMarket


class LinearEconomy(SimMarket, ABC):
    @staticmethod
    def get_competitor_classes() -> list:
        import recommerce.market.linear.linear_vendors as l_vendors
        return sorted(
            ut.filtered_class_str_from_dir('recommerce.market.linear.linear_vendors', dir(l_vendors), '.*LE.*Agent.*'))

    @staticmethod
    def get_configurable_fields() -> list:
        # TODO: reduce this list to only the required fields
        return [
            ('max_storage', int, greater_zero_rule),
            ('episode_length', int, greater_zero_rule),
            ('max_price', int, greater_zero_rule),
            ('max_quality', int, greater_zero_rule),
            ('number_of_customers', int, greater_zero_even_rule),
            ('production_price', int, non_negative_rule),
            ('storage_cost_per_product', (int, float), non_negative_rule),
            ('opposite_own_state_visibility', bool, None),
            ('common_state_visibility', bool, None),
            ('reward_mixed_profit_and_difference', bool, None),
            ('support_continuous_action_space', bool, None),
            ('fraction_of_strategic_customer', float, between_zero_one_rule),
            ('max_waiting_customers', int, non_negative_rule)
        ]

    def _setup_action_observation_space(self, support_continuous_action_space: bool) -> None:
        """
        Assume there are n vendors.
        The observation array for vendor i ∈ [0, ..., n - 1] has the following format:
        [ COMMON_STATE, PRICE_1, ... , PRICE_{n - 1}]

        The action space is discrete/continuous with as many actions as prices.

        Args:
            support_continuous_action_space (bool): If True, the action space will be continuous.
        """

        assert self.config.opposite_own_state_visibility, 'This market does not make sense without a visibility of the competitors own states.'  # noqa: E501

        if self.config.common_state_visibility:
            common_state_min = [
                0.0, # step counter
                0.0,  # waiting customers,
            ]

            common_state_max = [
                25,
                self.config.max_waiting_customers,
            ]

            min_observation_space = np.array(common_state_min + [0.0] * len(self.competitors), dtype=np.float32)
            max_observation_space = np.array(common_state_max + [self.config.max_price] * len(self.competitors), dtype=np.float32)
        else:
            min_observation_space = np.array([0.0] * len(self.competitors), dtype=np.float32)
            max_observation_space = np.array([self.config.max_price] * len(self.competitors), dtype=np.float32)

        self.observation_space = gym.spaces.Box(min_observation_space, max_observation_space)

        if support_continuous_action_space:
            self.action_space = gym.spaces.Box(np.array([0], dtype=np.float32), np.array([self.config.max_price], dtype=np.float32))
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
        return []

    def _choose_customer(self) -> Customer:
        return CustomerLinear()

    def _reset_vendor_actions(self) -> int:
        """
        Reset the price in the linear economy.
        Returns:
            int: The new price.
        """
        return self._convert_policy_value_into_action_space(self.config.production_price + 1)

    def _convert_policy_value_into_action_space(self, action_values):

        if not isinstance(action_values, np.ndarray):
            action_values = [action_values]

        if self.support_continuous_action_space:
            return np.array(action_values, dtype=np.float32)
        return action_values

    def _complete_purchase(self, profits, chosen_vendor, frequency, strategic=False) -> None:
        profits[chosen_vendor] += frequency * (self.vendor_actions[chosen_vendor] - self.config.production_price)
        if strategic:
            self._output_dict[f'customer/purchases_strategic'][f'vendor_{chosen_vendor}'] += frequency
        else:
            self._output_dict[f'customer/purchases'][f'vendor_{chosen_vendor}'] += frequency

    def _initialize_output_dict(self):
        self._ensure_output_dict_has('customer/purchases', [0] * self._number_of_vendors)
        self._ensure_output_dict_has('customer/purchases_strategic', [0] * self._number_of_vendors)
        self._ensure_output_dict_has('actions/price',
                                     [self.vendor_actions[i].item(0) if isinstance(self.vendor_actions[i], np.ndarray)
                                      else self.vendor_actions[i] for i in range(self._number_of_vendors)])

    def get_n_actions(self):
        return self.action_space.n

    def _is_probability_distribution_fitting_exactly(self, probability_distribution) -> bool:
        """
        The probability distribution must have one entry for buy_nothing and one entry (purchases_new) for every vendor.
        Args:
            probability_distribution (np.array): The probabilities that a customer either buys nothing or the new product of a specific vendor. # noqa: E501
        Returns:
            bool: Whether the probability_distribution fits into the LinearEcononmy.
        """
        return len(probability_distribution) == 1 + self._number_of_vendors

    def _get_common_state_array(self) -> ndarray:
        last_prices = list(self.price_deque)
        last_prices = np.pad(last_prices, (5 - len(last_prices), 0), 'constant')
        last_prices = []

        return np.array([
            self.step_counter % 25,
            self.waiting_customers,
          ])

    def _reset_common_state(self) -> None:
        self.waiting_customers = 0


class LinearEconomyMonopoly(LinearEconomy):
    @staticmethod
    def get_num_competitors() -> int:
        return 0

    def _get_competitor_list(self) -> list:
        return []


class LinearEconomyDuopoly(LinearEconomy):
    """
    This is a linear economy, with two vendors.
    """

    @staticmethod
    def get_num_competitors() -> int:
        return 1

    def _get_competitor_list(self) -> list:
        return [LinearRatio1LEAgent(config_market=self.config)]


class LinearEconomyOligopoly(LinearEconomy):
    """
    This is a linear economy, with multiple vendors.
    """

    @staticmethod
    def get_num_competitors() -> int:
        return np.inf

    def _get_competitor_list(self) -> list:
        return [
            LinearRatio1LEAgent(config_market=self.config),
            LERandomAgent(config_market=self.config),
            Just2PlayersLEAgent(config_market=self.config),
        ]
