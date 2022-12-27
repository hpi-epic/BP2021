import math
from abc import abstractmethod
from collections import deque
from statistics import mean
from typing import Tuple

import gym
import numpy as np
from attrdict import AttrDict

from recommerce.configuration.json_configurable import JSONConfigurable
from recommerce.configuration.utils import filtered_class_str_from_dir


# An offer is a market state that contains all prices and qualities

# There are three kinds of state:
# First: a common state for all vendors
# Second: a state specific to one vendor
# Third: vendor's actions from the former round which needs to be saved and influence the other's decision e.g. prices


class SimMarket(gym.Env, JSONConfigurable):
    """
    The superclass to all market environments.
    Abstract class that cannot be instantiated.
    Inherits from `gym.env`.
    """

    @staticmethod
    def get_num_competitors() -> int:
        raise NotImplementedError

    @staticmethod
    def get_possible_rl_agents() -> list:
        import recommerce.rl.actorcritic.actorcritic_agent as ac_agents
        import recommerce.rl.q_learning.q_learning_agent as q_agents
        from recommerce.rl.stable_baselines import sb_a2c, sb_ddpg, sb_ppo, sb_sac, sb_td3
        all_actorcritic = filtered_class_str_from_dir('recommerce.rl.actorcritic.actorcritic_agent', dir(ac_agents),
                                                      '^.*Agent.+|Discrete.+$')
        all_qlearning = filtered_class_str_from_dir('recommerce.rl.q_learning.q_learning_agent', dir(q_agents),
                                                    '^QLearningAgent$')
        all_stable_base_lines = filtered_class_str_from_dir('recommerce.rl.stable_baselines.sb_ddpg', dir(sb_ddpg),
                                                            '^StableBaselines(?!Agent).*')
        all_stable_base_lines += filtered_class_str_from_dir('recommerce.rl.stable_baselines.sb_a2c', dir(sb_a2c),
                                                             '^StableBaselines(?!Agent).*')
        all_stable_base_lines += filtered_class_str_from_dir('recommerce.rl.stable_baselines.sb_ppo', dir(sb_ppo),
                                                             '^StableBaselines(?!Agent).*')
        all_stable_base_lines += filtered_class_str_from_dir('recommerce.rl.stable_baselines.sb_sac', dir(sb_sac),
                                                             '^StableBaselines(?!Agent).*')
        all_stable_base_lines += filtered_class_str_from_dir('recommerce.rl.stable_baselines.sb_td3', dir(sb_td3),
                                                             '^StableBaselines(?!Agent).*')

        return sorted(all_actorcritic + all_qlearning + all_stable_base_lines)

    @staticmethod
    def get_competitor_classes() -> list:
        raise NotImplementedError

    def __init__(self, config: AttrDict, competitors: list = None) -> None:
        """
        Initialize a SimMarket instance.
        Set up needed values such as competitors and action/observation-space and reset the environment.
        By default, the marketplace supports discrete actions.
        You can activate continuous actions using setting support_continuous_action_space.

        Args:
            support_continuous_action_space (bool, optional): If True, the action space will be continuous. Defaults to False.
            competitors (list, optional): If not None, this overwrites the default competitor list with a custom one.
        """
        self.config = config
        self.support_continuous_action_space = self.config['support_continuous_action_space']
        self.competitors = self._get_competitor_list() if not competitors else competitors
        # The agent's price does not belong to the observation_space any more because an agent should not depend on it
        self._setup_action_observation_space(self.support_continuous_action_space)
        self._owner = None
        self._customer = None
        self._number_of_vendors = self._get_number_of_vendors()
        # TODO: Better testing for the observation and action space
        assert (self.observation_space and self.action_space), 'Your observation or action space is not defined'
        assert not self.config.reward_mixed_profit_and_difference or self._number_of_vendors > 1, \
            'You cannot use the mixed profit and difference reward in a monopoly market'
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
        self.price_deque = deque([], maxlen=5)

        self._reset_common_state()

        self.vendor_specific_state = [self._reset_vendor_specific_state() for _ in range(self._number_of_vendors)]
        self.vendor_actions = [self._reset_vendor_actions() for _ in range(self._number_of_vendors)]

        self._customer = self._choose_customer()
        self._owner = self._choose_owner()

        return self._observation()

    @abstractmethod
    def _convert_policy_value_into_action_space(self, probability_distribution) -> None:
        """
        Convert the received action values into the appropriate format. This prevents conversations warnings
        received from the gym environment
        """
        raise NotImplementedError

    @abstractmethod
    def _is_probability_distribution_fitting_exactly(self, probability_distribution) -> None:
        """
        The implementation of this function varies between economy types.
        See also:
            `<market.linear_sim_market.LinearEconomy._is_probability_distribution_fitting_exactly`
            `<market.circular.circular_sim_market.CircularEconomy._is_probability_distribution_fitting_exactly>`
        """
        raise NotImplementedError

    def _simulate_customers_level_1(self, profits, number_of_customers) -> None:

        # first split customer into myopic and recurring
        p_recurr = 0.3

        number_of_recurring_customer = np.random.binomial(number_of_customers, p_recurr)
        number_of_myopic_customer = number_of_customers - number_of_recurring_customer

        # add recurring customer
        number_of_returning_recurring_customer = math.floor(self.waiting_customers * 0.3)
        self.waiting_customers = max(0, self.waiting_customers - number_of_returning_recurring_customer)
        number_of_recurring_customer += number_of_returning_recurring_customer

        # track incoming customer stream
        self._output_dict['customer/incoming'] += number_of_recurring_customer + number_of_myopic_customer

        # get probability distributions
        probability_distribution = self._customer.generate_purchase_probabilities_from_offer(
            self.step_counter, self._get_common_state_array(), self.vendor_specific_state, self.vendor_actions)
        assert isinstance(probability_distribution,
                          np.ndarray), 'generate_purchase_probabilities_from_offer must return an np.ndarray'
        assert self._is_probability_distribution_fitting_exactly(probability_distribution)

        # simulate myopic customers
        myopic_customer_decisions = np.random.multinomial(number_of_myopic_customer, probability_distribution).tolist()
        self._output_dict['customer/buy_nothing'] += myopic_customer_decisions[0]
        for seller, frequency in enumerate(myopic_customer_decisions):
            if seller == 0 or frequency == 0:
                continue
            self._complete_purchase(profits, seller - 1, frequency)

        # simulate recurring customers
        recurring_customer_decision = np.random.multinomial(number_of_recurring_customer,
                                                            probability_distribution).tolist()
        dont_buy = recurring_customer_decision[0]
        enter_waiting = math.floor(dont_buy * 0.9)
        self.waiting_customers = min(self.waiting_customers + enter_waiting, self.config.max_waiting_customers)
        self._output_dict['customer/buy_nothing'] += (dont_buy - enter_waiting)

        for seller, frequency in enumerate(recurring_customer_decision):
            if seller == 0 or frequency == 0:
                continue
            self._complete_purchase(profits, seller - 1, frequency)

    def _simulate_customers(self, profits, number_of_customers) -> None:
        """
        Simulate the customers, the products offered by the vendors get sold to n customers.
        For the offers, the internal state is used.
        The profits for each vendor get saved to the profits array.
        Args:
            profits (np.array): The profits of the customers get saved to this array
            number_of_customers (int): the number of customers eager to buy each step.
        """

        number_of_myopic_customer = number_of_customers * (1 - self.config.fraction_of_strategic_customer)
        number_of_strategic_customer = number_of_customers - number_of_myopic_customer

        # if we don't have enough observations, there won't be any strategic customers
        if len(self.price_deque) < 5:
            number_of_myopic_customer = number_of_customers
            number_of_strategic_customer = 0

        self._output_dict['customer/incoming'] += number_of_myopic_customer

        probability_distribution = self._customer.generate_purchase_probabilities_from_offer(
            self.step_counter, self._get_common_state_array(), self.vendor_specific_state, self.vendor_actions)
        assert isinstance(probability_distribution,
                          np.ndarray), 'generate_purchase_probabilities_from_offer must return an np.ndarray'
        assert self._is_probability_distribution_fitting_exactly(probability_distribution)

        customer_decisions = np.random.multinomial(number_of_myopic_customer, probability_distribution).tolist()

        self._output_dict['customer/buy_nothing'] += customer_decisions[0]
        for seller, frequency in enumerate(customer_decisions):
            if seller == 0 or frequency == 0:
                continue
            self._complete_purchase(profits, seller - 1, frequency)

        self._simulate_strategic_customer(profits, number_of_strategic_customer)

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        """
        Simulate the market between actions by the agent.
        It is part of the gym library for reinforcement learning.
        It is pretty generic and configured by overwriting the abstract and empty methods.
        Args:
            action (int | Tuple): The action of the agent. In discrete case: the action must be between 0 and number of actions -1.
            Note that you must add one to this price to get the real price!
        Returns:
            Tuple[np.array, float, bool, dict]: A Tuple,
            containing the observation the agents makes right before his next action,
            the reward he made between these actions,
            a flag indicating if the market closes and information about the market for logging purposes.
        """
        if isinstance(action, np.ndarray) and len(action) == 1 and not self.config.support_continuous_action_space:
            action = action.item()

        action = self._convert_policy_value_into_action_space(action)
        assert self.action_space.contains(action), f'{action} ({type(action)}) invalid'

        self.vendor_actions[0] = action

        self.step_counter += 1

        profits = [0] * self._number_of_vendors

        self._output_dict = {'customer/buy_nothing': 0, 'customer/incoming': 0}

        self._initialize_output_dict()

        customers_per_vendor_iteration = self.config.number_of_customers // self._number_of_vendors
        for i in range(self._number_of_vendors):
            self._simulate_customers(profits, customers_per_vendor_iteration)

            if i == 0:
                self.price_deque.append(self.vendor_actions[0])
            else:
                if self.price_deque[-1] > self.vendor_actions[i]:
                    self.price_deque.pop()
                    self.price_deque.append(self.vendor_actions[i])

            if self._owner is not None:
                self._simulate_owners(profits)

            # the competitor, which turn it is, will update its pricing
            if i < len(self.competitors):
                action_competitor_i = self._convert_policy_value_into_action_space(
                    self.competitors[i].policy(self._observation(i + 1)))
                assert self.action_space.contains(action_competitor_i), \
                    f'This vendor does not deliver a suitable action, action_space: {self.action_space}, action: {action_competitor_i}'
                self.vendor_actions[i + 1] = action_competitor_i

        self._consider_storage_costs(profits)

        self._ensure_output_dict_has('profits/all', profits)

        self._output_dict['customers/waiting'] = self.waiting_customers

        is_done = self.step_counter >= self.config.episode_length

        if not self.config.reward_mixed_profit_and_difference:
            reward = profits[0]
        else:
            reward = 2 * profits[0] - np.max(profits[1:])

        return self._observation(), float(reward), is_done, self._output_dict

    def _simulate_strategic_customer(self, profits, number_of_new_strategic_customer):
        if len(self.price_deque) < 5:
            return

        number_of_reoccurring_strategic_customer = math.floor(self.waiting_customers * 0.3)
        number_of_strategic_customer = number_of_reoccurring_strategic_customer + number_of_new_strategic_customer
        self._output_dict[
            'customer/incoming'] += number_of_strategic_customer  # this includes new + returning strategic customer
        self.waiting_customers = max(self.waiting_customers - number_of_reoccurring_strategic_customer, 0)

        if number_of_strategic_customer == 0:
            return

        current_lowest_offer_price_vendor, current_lowest_offer_price = min(enumerate(self.vendor_actions),
                                                                            key=lambda x: x[1])
        avg_price = sum(self.price_deque) / len(self.price_deque)

        if current_lowest_offer_price < 0.85 * avg_price:
            self._complete_purchase(profits, current_lowest_offer_price_vendor, number_of_strategic_customer)
        else:
            # 90% of the strategic customers who couldn't purchase will enter waiting state
            if self.config.fraction_of_strategic_customer > 0:
                self.waiting_customers = min(self.waiting_customers + (max(int(number_of_strategic_customer * 0.9), 1)),
                                             self.config.max_waiting_customers)

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
        # observatons is the array containing the global states. We add everything relevant to it, then return a concatenated version.
        observations = [self._get_common_state_array()] if self.config.common_state_visibility else []
        if self.config.common_state_visibility:
            assert isinstance(observations[0], np.ndarray), '_get_common_state_array must return an np.ndarray'

        # first the state of the vendor whose view we create will be added
        if self.vendor_specific_state[vendor_view] is not None:
            observations.append(np.array(self.vendor_specific_state[vendor_view], ndmin=1, dtype=np.float32))

        # the rest of the vendors actions and states will be added
        for vendor_index in range(self._number_of_vendors):
            if vendor_index == vendor_view:
                continue
            observations.append(np.array(self.vendor_actions[vendor_index], ndmin=1, dtype=np.float32))
            if self.vendor_specific_state[vendor_index] is not None and self.config.opposite_own_state_visibility:
                observations.append(np.array(self.vendor_specific_state[vendor_index], ndmin=1, dtype=np.float32))

        # The observation has to be part of the observation_space defined by the market
        concatenated_observations = np.concatenate(observations, dtype=np.float32)
        assert self.observation_space.contains(concatenated_observations), \
            f'{concatenated_observations} ({type(concatenated_observations)}) invalid observation'
        return concatenated_observations

    def _reset_common_state(self) -> None:
        pass

    @abstractmethod
    def _get_common_state_array(self) -> None:
        """
        The implementation of this function varies between economy types.
        See also:
            `<market.linear.linear_sim_market.LinearEconomy._get_common_state_array>`
            `<market.circular.circular_sim_market.CircularEconomy._get_common_state_array>`
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_vendor_specific_state(self) -> None:
        """
        The implementation of this function varies between economy types.
        See also:
            `<market.linear.linear_sim_market.LinearEconomy._reset_vendor_specific_state>`
            `<market.circular.circular_sim_market.CircularEconomy._reset_vendor_specific_state>`
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
    def _setup_action_observation_space(self, support_continuous_action_space) -> None:  # pragma: no cover
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

    def get_observations_dimension(self) -> int:
        """
        Get the dimension of the observation space.
        This can be used to set the number of inputs for vendors.

        Returns:
            int: The dimension of the observation space.
        """
        return self.observation_space.shape[0]

    def get_actions_dimension(self) -> int:
        """
        Get the dimension of the action space.
        This can be used to set the number of outputs for vendors with continuous action space.

        Returns:
            int: The dimension of the action space.
        """
        return 1 if self.action_space.shape is not None else len(self.action_space)

    @abstractmethod
    def _get_competitor_list(self) -> list:  # pragma: no cover
        """
        Get a list of all competitors in the current market scenario.

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

    def _ensure_output_dict_has(self, name, init_for_all_vendors=None) -> None:
        """
        Ensure that the _output_dict has an entry with the given name and create an entry otherwise.
        If a parameter for init_for_all_vendors is passed, it will be interpreted as creating a dict with the passed array as content.

        Args:
            name (string): name of the dict entry which should be checked.
            init_for_all_vendors (list, optional): initialization values for all vendors in this entry. Defaults to None.
        """
        if init_for_all_vendors is not None:
            assert isinstance(init_for_all_vendors, list) and len(init_for_all_vendors) == self._number_of_vendors, \
                'make sure you pass a list with length of number of vendors'
        if name not in self._output_dict:
            if init_for_all_vendors is None:
                self._output_dict[name] = 0
            else:
                self._output_dict[name] = dict(
                    zip([f'vendor_{i}' for i in range(self._number_of_vendors)], init_for_all_vendors))

    @abstractmethod
    def get_configurable_fields() -> list:
        """
        Return a list of keys that can be used to configure this marketplace using a `market_config.json`.
        Also contains key types and validation logic.

        Returns:
            list: The list of (key, type, validation).
        """
        raise NotImplementedError
