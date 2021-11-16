#!/usr/bin/env python3

import math

import gym
import numpy as np

import competitor as comp
import utils
from customer import CustomerLinear

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
        return self.state

    def simulate_customers(self, profits, customer_information, n):
        for _ in range(n):
            customer_action = self.customer.buy_object(customer_information)
            if customer_action != 0:
                profits[customer_action - 1] += (
                    customer_information[(customer_action - 1) * 2]
                    - utils.PRODUCTION_PRICE
                )

    def full_view(self, action):
        return np.concatenate(
            (self.action_to_array(action), self.state), dtype=np.float64
        )

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
                self.full_view(action),
                math.floor(utils.NUMBER_OF_CUSTOMERS / n_vendors),
            )
            if i < len(self.competitors):
                act_compet_i = self.competitors[i].give_competitors_price(
                    self.full_view(action), i + 1
                )
                self.apply_compet_action(act_compet_i, i)

        output_dict = {'all_profits': profits}
        is_done = self.counter >= utils.EPISODE_LENGTH
        return self.state, profits[0], is_done, output_dict


class LinearEconomy(SimMarket):
    def setup_act_obs_space(self):
        # cell 0: agent's quality, afterwards: odd cells: competitor's price, even cells: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0] * (len(self.competitors) * 2 + 1)),
            np.array(
                [utils.MAX_QUALITY]
                + [utils.MAX_PRICE, utils.MAX_QUALITY] * len(self.competitors)
            ),
            dtype=np.float64,
        )

        # one action for every price possible - 2 for 0 and MAX_PRICE
        self.action_space = gym.spaces.Discrete(utils.MAX_PRICE)

    def reset_agent_information(self):
        return [utils.shuffle_quality()]

    def reset_competitor_information(self, competitor):
        comp_price, comp_quality = competitor.reset()
        return [comp_price, comp_quality]

    def action_to_array(self, action):
        return np.array([action + 1.0])

    def choose_customer(self):
        return CustomerLinear()

    def ith_compet_index(self, i):
        return 2 * i + 1

    def apply_compet_action(self, action, i):
        self.state[self.ith_compet_index(i)] = action


class ClassicScenario(LinearEconomy):
    def get_competitor_list(self):
        return [comp.CompetitorLinearRatio1()]


class MultiCompetitorScenario(LinearEconomy):
    def get_competitor_list(self):
        return [
            comp.CompetitorLinearRatio1(),
            comp.CompetitorRandom(),
            comp.CompetitorJust2Players(),
        ]


class CircularEconomy:
    pass
