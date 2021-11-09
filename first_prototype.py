import gym
import numpy as np
import random
import utils
from competitor import Competitor

# An offer is a Market State that contains both prices and both qualities


def buy_object(offers):
    if random.random() < 0.17:
        return random.randint(1, 2)
    value_agent = max(offers[1] / offers[0] + np.random.normal() / 2, 0.1)
    value_compet = max(offers[3] / offers[2] + np.random.normal() / 2, 0.1)
    maxprice = np.random.normal() * 3 + 25
    if offers[0] > maxprice:
        value_agent = 0
    if offers[2] > maxprice:
        value_compet = 0

    if value_agent == 0 and value_compet == 0:
        return 0  # Don't buy anything
    elif value_agent > value_compet:
        return 1  # Buy agent's
    else:
        return 2  # Buy competitor's


class SimMarket(gym.Env):

    def __init__(self):
        utils.setup()

        self.competitor = Competitor()
        # cell 0: agent's price, cell 1: agent's quality, cell 2: competitor's price, cell 3: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0, 0.0, 0.0, 0.0]), np.array([utils.MAX_PRICE, utils.MAX_QUALITY, utils.MAX_PRICE, utils.MAX_QUALITY]), dtype=np.float64)
        # one action for every price possible - 2 for 0 and MAX_PRICE
        self.action_space = gym.spaces.Discrete(utils.MAX_PRICE - 2 )

    def shuffle_quality(self):
        return min(max(int(np.random.normal(utils.MAX_QUALITY/2, utils.MAX_QUALITY/5)), 1), utils.MAX_QUALITY)

    def reset(self, random_start=True):
        self.counter = 0
        self.comp_profit_overall = 0
        if random_start == False:
            random_start = random.random() < 0.5
        
        agent_price = int(utils.PRODUCTION_PRICE +
                          np.random.normal(3,3))if random_start else utils.PRODUCTION_PRICE
        agent_quality = self.shuffle_quality()

        comp_price, comp_quality = self.competitor.reset(random_start)
        
        self.state = np.array(
            [agent_price, agent_quality, comp_price, comp_quality])
        print("I initiate with ", self.state)
        return self.state[0:4]

    def step(self, action):

        err_msg = "%r (%s) invalid" % (action, type(action))
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
        for _ in range(10):
            customer_action = buy_object(self.state)
            if customer_action == 1:
                profit_agent += self.state[0] - utils.PRODUCTION_PRICE
                agent_sales += 1
            elif customer_action == 2:
                comp_profit += self.state[2] - utils.PRODUCTION_PRICE
                comp_sales += 1

        self.state[2] = self.competitor.give_competitors_price(self.state)
        self.state[2] = max(1, self.state[2])

        for _ in range(10):
            customer_action = buy_object(self.state)
            if customer_action == 1:
                profit_agent += self.state[0] - utils.PRODUCTION_PRICE
                agent_sales += 1
            elif customer_action == 2:
                comp_profit += self.state[2] - utils.PRODUCTION_PRICE
                comp_sales += 1

        # print("You sold " + str(agent_sales) +
        #       " and your competitor " + str(comp_sales))
        # print('comp profit this round is', comp_profit)
        output_dict = {
            'comp_profit': comp_profit
        }
        is_done = self.counter >= utils.STEPS_PER_ROUND
        return self.state[0:4], profit_agent, is_done, output_dict
