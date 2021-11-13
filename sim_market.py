#!/usr/bin/env python3

import random

import gym
import numpy as np

import competitor as comp
import utils
from customer import Customer

# An offer is a Market State that contains both prices and both qualities


class SimMarket(gym.Env):
    def __init__(self):
        self.competitors = [comp.CompetitorLinearRatio1()]
        # The agent's price does not belong to the observation_space any more because an agent should not depend on it
        # cell 0: agent's quality, cell 1: competitor's price, cell 2: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0, 0.0, 0.0]),
            np.array([utils.MAX_QUALITY, utils.MAX_PRICE, utils.MAX_QUALITY]),
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

        profits = [0, 0]  # [0] * (len(self.competitors) + 1)
        mycustomer = Customer()

        # self.simulate_customers(
        #     profits,
        #     self.full_view(action),
        #     int(utils.NUMBER_OF_CUSTOMERS / 2),
        #     mycustomer,
        # )
        # self.state[1] = self.competitors[0].give_competitors_price(self.full_view(action), 1)
        # self.simulate_customers(
        #     profits,
        #     self.full_view(action),
        #     int(utils.NUMBER_OF_CUSTOMERS / 2),
        #     mycustomer,
        # )

        for i in range(len(self.competitors) + 1):
            self.simulate_customers(
                profits,
                self.full_view(action),
                int(utils.NUMBER_OF_CUSTOMERS / 2),
                mycustomer,
            )
            if i < len(self.competitors):
                self.state[2 * (i + 1) - 1] = self.competitors[
                    i
                ].give_competitors_price(self.full_view(action), 1)

        output_dict = {'comp_profit': profits[1], 'all_profits': profits}
        is_done = self.counter >= utils.EPISODE_LENGTH
        return self.state, profits[0], is_done, output_dict
