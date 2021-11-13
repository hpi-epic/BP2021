#!/usr/bin/env python3

# helper
import random

# rl
import gym
import numpy as np

# own files
import utils
from competitor import Competitor
from customer import Customer

# An offer is a Market State that contains both prices and both qualities


class SimMarket(gym.Env):
    def __init__(self):
        self.competitor = Competitor()
        # The agent's price does not belong to the observation_space any more because an agent should not depend on it
        # cell 0: agent's quality, cell 1: competitor's price, cell 2: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0, 0.0, 0.0]),
            np.array([utils.MAX_QUALITY, utils.MAX_PRICE, utils.MAX_QUALITY]),
            dtype=np.float64,
        )

        # one action for every price possible - 2 for 0 and MAX_PRICE
        self.action_space = gym.spaces.Discrete(utils.MAX_PRICE - 2)

    def reset(self, random_start=True):
        self.counter = 0
        self.comp_profit_overall = 0
        if not random_start:
            random_start = random.random() < 0.5

        agent_quality = utils.shuffle_quality()

        comp_price, comp_quality = self.competitor.reset(random_start)

        self.state = np.array([agent_quality, comp_price, comp_quality])
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
        np.concatenate((np.array([action + 1.0]), self.state), dtype=np.float64)

    def step(self, action):
        # The action is the new price of the agent

        err_msg = '%r (%s) invalid' % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.counter += 1

        profits = [0, 0]
        mycustomer = Customer()
        self.simulate_customers(
            profits,
            self.full_view(action),
            int(utils.NUMBER_OF_CUSTOMERS / 2),
            mycustomer,
        )
        self.state[1] = self.competitor.give_competitors_price(self.full_view(action))
        self.simulate_customers(
            profits,
            self.full_view(action),
            int(utils.NUMBER_OF_CUSTOMERS / 2),
            mycustomer,
        )

        output_dict = {'comp_profit': profits[1]}
        is_done = self.counter >= utils.EPISODE_LENGTH
        return self.state, profits[0], is_done, output_dict
