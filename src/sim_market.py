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
        # cell 0: agent's price, cell 1: agent's quality, cell 2: competitor's price, cell 3: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array(
                [utils.MAX_PRICE, utils.MAX_QUALITY, utils.MAX_PRICE, utils.MAX_QUALITY]
            ),
            dtype=np.float64,
        )

        # one action for every price possible - 2 for 0 and MAX_PRICE
        self.action_space = gym.spaces.Discrete(utils.MAX_PRICE - 2)

    def shuffle_quality(self):
        return min(
            max(int(np.random.normal(utils.MAX_QUALITY / 2, utils.MAX_QUALITY / 5)), 1),
            utils.MAX_QUALITY,
        )

    def reset(self, random_start=True):
        self.counter = 0
        self.comp_profit_overall = 0
        if random_start is False:
            random_start = random.random() < 0.5

        agent_price = (
            int(utils.PRODUCTION_PRICE + np.random.normal(3, 3))
            if random_start
            else utils.PRODUCTION_PRICE
        )
        agent_quality = self.shuffle_quality()

        comp_price, comp_quality = self.competitor.reset(random_start)

        self.state = np.array([agent_price, agent_quality, comp_price, comp_quality])
        print('I initiate with', self.state)
        return self.state[0:4]

    def step(self, action):

        err_msg = '%r (%s) invalid' % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.counter += 1

        # The action is the new price of the agent
        self.state[0] = action
        self.state[0] = max(1, self.state[0])

        if self.state[0] >= utils.MAX_PRICE:
            self.state[0] = utils.MAX_PRICE - 1
            print(self.state)
            return self.state, -1000, self.counter >= utils.STEPS_PER_ROUND, {}

        profit_agent = 0
        comp_profit = 0
        agent_sales = 0
        comp_sales = 0

        for iter in range(2):

            for _ in range(int(utils.NUMBER_OF_CUSTOMERS / 2)):
                customer_action = Customer.buy_object(self.state)
                if customer_action == 1:
                    profit_agent += self.state[0] - utils.PRODUCTION_PRICE
                    agent_sales += 1
                elif customer_action == 2:
                    comp_profit += self.state[2] - utils.PRODUCTION_PRICE
                    comp_sales += 1
            # calculate the new price of the competitor
            if iter == 0:
                self.state[2] = max(
                    1, self.competitor.give_competitors_price(self.state)
                )

        # print('You sold ' + str(agent_sales) +
        #       ' and your competitor ' + str(comp_sales))
        # print('comp profit this round is', comp_profit)
        output_dict = {'comp_profit': comp_profit}
        is_done = self.counter >= utils.STEPS_PER_ROUND
        return self.state[0:4], profit_agent, is_done, output_dict
