import gym
import math
import numpy as np
import random
from competitor import Competitor
import utils

# An offer is a Market State that contains both prices and both qualities


def buy_object_old(offers):
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

def softmax(priorities):
    exp_priorities = np.exp(priorities)
    return exp_priorities / sum(exp_priorities)

def probabilities(offers):
    value_agent = (offers[1] + 20) / offers[0] - math.exp(offers[0] - 25)
    value_compet = (offers[3] + 20) / offers[2] - math.exp(offers[2] - 25)
    buy_nothing = max(0, min(offers[1], offers[2]) - 23) + 1
    return softmax([value_agent, value_compet, buy_nothing])

def buy_object(offers):
    probs = probabilities(offers)
    myrand = random.random()
    if myrand < probs[0]:
        return 1
    elif myrand < probs[0] + probs[1]:
        return 2
    else:
        return 0

class SimMarket(gym.Env):
    STEPS_PER_ROUND = 50

    def __init__(self, maxprice=30.0, maxquality=10.0):
        self.maxprice = maxprice
        self.maxquality = maxquality
        self.competitor = Competitor()
        # cell 0: agent's price, cell 1: agent's quality, cell 2: competitor's price, cell 3: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0, 0.0, 0.0, 0.0]), np.array([self.maxprice, self.maxquality, self.maxprice, self.maxquality]), dtype=np.float64)
        # 0: Decrease the price by 1, 1: keep the price constant, 2: decrease the price by 1
        self.action_space = gym.spaces.Discrete(28)

    def reset(self, random_start=True):
        self.counter = 0
        self.comp_profit_overall = 0
        # The production price is initially set to maxprice / 3 for simplicity reasons
        self.production_price = int(self.maxprice / 3)
        # The agent's quality is set to fixed maxquality / 2
        if random_start == False:
            random_start = random.random() < 0.5
        
        agent_price = int(self.production_price +
                          np.random.normal() * 3 + 3)if random_start else 10
        agent_quality = utils.shuffle_quality()

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

        if self.state[0] >= self.maxprice:
            self.state[0] = self.maxprice - 1
            print(self.state)
            return self.state, -1000, self.counter >= self.STEPS_PER_ROUND, {}

        profit_agent = 0
        comp_profit = 0
        agent_sales = 0
        comp_sales = 0
        for _ in range(10):
            customer_action = buy_object(self.state)
            if customer_action == 1:
                profit_agent += self.state[0] - self.production_price
                agent_sales += 1
            elif customer_action == 2:
                comp_profit += self.state[2] - self.production_price
                # self.comp_profit_overall = comp_profit
                comp_sales += 1

        self.state[2] = self.competitor.give_competitors_price(self.state)
        self.state[2] = max(1, self.state[2])

        for _ in range(10):
            customer_action = buy_object(self.state)
            if customer_action == 1:
                profit_agent += self.state[0] - self.production_price
                agent_sales += 1
            elif customer_action == 2:
                comp_profit += self.state[2] - self.production_price
                # self.comp_profit_overall = comp_profit
                comp_sales += 1

        # print("You sold " + str(agent_sales) +
        #       " and your competitor " + str(comp_sales))
        # print('comp profit this round is', comp_profit)
        output_dict = {
            'comp_profit': comp_profit
        }
        is_done = self.counter >= self.STEPS_PER_ROUND
        return self.state[0:4], profit_agent, is_done, output_dict
