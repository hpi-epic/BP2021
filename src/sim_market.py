#!/usr/bin/env python3

import copy

import gym
import numpy as np

import competitor as comp
import customer
import utils as ut

# An offer is a Market State that contains both prices and both qualities


class SimMarket(gym.Env):
    def __init__(self):
        self.competitors = self.get_competitor_list()
        # The agent's price does not belong to the observation_space any more because an agent should not depend on it
        self.setup_act_obs_space()

        # TODO: Better testing for the observation and action space
        assert (
            self.observation_space and self.action_space
        ), 'Your subclass has major problems with setting up the environment'

    def reset(self):
        self.counter = 0

        tmpstate = self.reset_agent_information()
        for c in self.competitors:
            tmpstate += self.reset_competitor_information(c)

        self.state = np.array(tmpstate)

        self.customer = self.choose_customer()

        print('I initiate with', self.state)
        return copy.deepcopy(self.state)

    def simulate_customers(self, profits, offers, n):
        for _ in range(n):
            customer_buy, customer_return = self.customer.buy_object(offers)
            if customer_return is not None:
                self.apply_customer_return(customer_return)
            if customer_buy != 0:
                self.complete_purchase(offers, profits, customer_buy)

    def generate_offer(self, action):
        return np.concatenate(
            (self.action_to_array(action), self.state), dtype=np.float64
        )

    def modify_profit_by_state(self, profits):
        pass

    def apply_customer_return(self, customer_return):
        assert False

    def step(self, action):
        # The action is the new price of the agent

        err_msg = '%r (%s) invalid' % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.counter += 1
        n_vendors = (
            len(self.competitors) + 1
        )  # The number of competitors plus the agent

        profits = [0] * n_vendors

        for i in range(n_vendors):
            self.simulate_customers(
                profits,
                self.generate_offer(action),
                int(np.floor(ut.NUMBER_OF_CUSTOMERS / n_vendors)),
            )
            if i < len(self.competitors):
                act_compet_i = self.competitors[i].give_competitors_price(
                    self.generate_offer(action), i + 1
                )
                self.apply_compet_action(act_compet_i, i)

        self.modify_profit_by_state(profits)

        output_dict = {'all_profits': profits}
        is_done = self.counter >= ut.EPISODE_LENGTH
        return copy.deepcopy(self.state), profits[0], is_done, output_dict


class LinearEconomy(SimMarket):
    def setup_act_obs_space(self):
        # cell 0: agent's quality, afterwards: odd cells: competitor's price, even cells: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0] * (len(self.competitors) * 2.0 + 1.0)),
            np.array(
                [ut.MAX_QUALITY]
                + [ut.MAX_PRICE, ut.MAX_QUALITY] * len(self.competitors)
            ),
            dtype=np.float64,
        )

        # one action for every price possible - 2 for 0 and MAX_PRICE
        self.action_space = gym.spaces.Discrete(ut.MAX_PRICE)

    def reset_agent_information(self):
        return [ut.shuffle_quality()]

    def reset_competitor_information(self, competitor):
        comp_price, comp_quality = competitor.reset()
        return [comp_price, comp_quality]

    def action_to_array(self, action):
        return np.array([action + 1.0])

    def choose_customer(self):
        return customer.CustomerLinear()

    def complete_purchase(self, offers, profits, customer_buy):
        profits[customer_buy - 1] += (offers[(customer_buy - 1) * 2] - ut.PRODUCTION_PRICE)

    def ith_compet_index(self, i):
        return 2 * i + 1

    def apply_compet_action(self, action, i):
        self.state[self.ith_compet_index(i)] = action


class ClassicScenario(LinearEconomy):
    def get_competitor_list(self):
        return [comp.CompetitorJust2Players()]


class MultiCompetitorScenario(LinearEconomy):
    def get_competitor_list(self):
        return [
            comp.CompetitorLinearRatio1(),
            comp.CompetitorRandom(),
            comp.CompetitorJust2Players(),
        ]


class CircularEconomy(SimMarket):
    def setup_act_obs_space(self):
        # cell 0: number of products in the used storage, cell 1: number of products in circulation
        self.observation_space = gym.spaces.Box(np.array([0, 0]), np.array([ut.MAX_STORAGE, 10 * ut.MAX_STORAGE]), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(ut.MAX_PRICE * ut.MAX_PRICE)  # Every pair of actions encoded in one number

    def get_competitor_list(self):
        return []

    def reset_agent_information(self):
        return [int(np.random.rand() * ut.MAX_STORAGE), int(5 * np.random.rand() * ut.MAX_STORAGE)]

    def action_to_array(self, action):
        # cell 0: price for second-hand-product, cell 1: price for new product
        act = [int(np.floor(action / ut.MAX_PRICE)) + 1, int(action % ut.MAX_PRICE) + 1]
        # print("You perform ", act)
        return act

    def choose_customer(self):
        return customer.CustomerCircular()

    def apply_customer_return(self, customer_return):
        assert customer_return == 1
        # print("A customer returns a product")
        if self.state[1] >= customer_return:
            if self.state[0] < ut.MAX_STORAGE:
                self.state[0] += customer_return
            self.state[1] -= customer_return

    def complete_purchase(self, offers, profits, customer_buy):
        # print("I want to buy ", customer_buy)
        assert len(profits) == 1
        assert 0 < customer_buy and customer_buy <= 2
        if customer_buy == 1:
            if self.state[0] >= 1:
                # Increase the profit and decrease the storage
                profits[0] += offers[0]
                self.state[0] -= 1
            else:
                # Punish the agent for not having enough second-hand-products
                profits[0] -= 2 * ut.MAX_PRICE
        elif customer_buy == 2:
            profits[0] += offers[1] - ut.PRODUCTION_PRICE
            # One more product is in circulation now
            self.state[1] = min(self.state[1] + 1, 10 * ut.MAX_STORAGE)

    def modify_profit_by_state(self, profits):
        # print("Your storage cost is ", self.state[0])
        profits[0] -= self.state[0] / 2  # Storage costs per timestep
