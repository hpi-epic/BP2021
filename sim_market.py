#!/usr/bin/env python3

import math
import random

import gym
import numpy as np

import competitor as comp
import utils
from customer import CustomerLinear

# An offer is a Market State that contains both prices and both qualities


class SimMarket(gym.Env):
    def __init__(self):
        self.competitors = [
            comp.CompetitorLinearRatio1(),
            comp.CompetitorRandom(),
            comp.CompetitorJust2Players(),
        ]
        # The agent's price does not belong to the observation_space any more because an agent should not depend on it
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

    def reset(self, random_start=True):
        self.counter = 0
        self.comp_profit_overall = 0
        if not random_start:
            random_start = random.random() < 0.5

        agent_quality = utils.shuffle_quality()

        tmpstate = [agent_quality]
        for c in self.competitors:
            comp_price, comp_quality = c.reset(random_start)
            tmpstate.append(comp_price)
            tmpstate.append(comp_quality)

        self.state = np.array(tmpstate)
        print('I initiate with', self.state)
        return self.state

    def simulate_customers(self, profits, customer_information, n, mycustomer):
        for _ in range(n):
            customer_action = mycustomer.buy_object(customer_information)
            if customer_action != 0:
                profits[customer_action - 1] += (
                    customer_information[(customer_action - 1) * 2]
                    - utils.PRODUCTION_PRICE
                )

    def full_view(self, action):
        return np.concatenate((np.array([action + 1.0]), self.state), dtype=np.float64)

    def step(self, action):
        # The action is the new price of the agent

        err_msg = '%r (%s) invalid' % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.counter += 1
        n_vendors = len(self.competitors) + 1

        profits = [0] * n_vendors
        mycustomer = CustomerLinear()

        for i in range(n_vendors):
            self.simulate_customers(
                profits,
                self.full_view(action),
                math.floor(utils.NUMBER_OF_CUSTOMERS / n_vendors),
                mycustomer,
            )
            if i < len(self.competitors):
                self.state[2 * (i + 1) - 1] = self.competitors[
                    i
                ].give_competitors_price(self.full_view(action), 1)

        output_dict = {'all_profits': profits}
        is_done = self.counter >= utils.EPISODE_LENGTH
        return self.state, profits[0], is_done, output_dict
