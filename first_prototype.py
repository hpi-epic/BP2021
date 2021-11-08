import random

import gym
import numpy as np

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
    def __init__(self, maxprice=30.0, maxquality=100.0):
        self.maxprice = maxprice
        self.maxquality = maxquality
        # cell 0: agent's price, cell 1: agent's quality, cell 2: competitor's price, cell 3: competitor's quality
        self.observation_space = gym.spaces.Box(
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([self.maxprice, self.maxquality, self.maxprice, self.maxquality]),
            dtype=np.float64,
        )
        # 0: Decrease the price by 1, 1: keep the price constant, 2: decrease the price by 1
        self.action_space = gym.spaces.Discrete(3)

    def give_competitors_action(self):
        ratio = (self.state[1] / self.state[0]) / (self.state[3] / self.state[2])
        if random.random() < 0.29:
            return random.randint(1, 2)
        elif self.state[2] < self.production_price or (
            ratio < 0.95 and self.state[2] < self.maxprice - 5
        ):
            # print("I increase with state ", self.state[2])
            return 2
        elif (
            self.state[2] > self.production_price + 1
            and ratio > 1.1
            or self.state[2] > self.maxprice - 5
        ):
            return 0
        else:
            return 1

    def shuffle_quality(self):
        return min(max(int(np.random.normal(50, 20)), 1), self.maxquality)

    def reset(self, randomstart=True):
        self.counter = 0
        self.comp_profit = 0
        # The production price is initially set to maxprice / 3 for simplicity reasons
        self.production_price = int(self.maxprice / 3)
        # The agent's quality is set to fixed maxquality / 2
        if not randomstart:
            randomstart = random.random() < 0.5
        self.state = np.array(
            [
                int(self.production_price + np.random.normal() * 3 + 3)
                if randomstart
                else 10,
                self.shuffle_quality(),
                int(self.production_price + np.random.normal() * 3 + 3)
                if randomstart
                else 10,
                self.shuffle_quality(),
            ]
        )
        print('I initiate with ', self.state)
        return self.state

    def step(self, action):
        err_msg = '%r (%s) invalid' % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.counter += 1
        competitors_action = self.give_competitors_action()
        if competitors_action == 0:
            self.state[2] -= 1
        elif competitors_action == 2:
            self.state[2] += 1

        if action == 0:
            self.state[0] -= 1
        elif action == 2:
            self.state[0] += 1

        self.state[0] = max(1, self.state[0])
        self.state[2] = max(1, self.state[2])

        if self.state[0] >= self.maxprice:
            self.state[0] = self.maxprice - 1
            print(self.state)
            return self.state, -1000, self.counter >= 50, {}

        profit_agent = 0
        agent_sales = 0
        comp_sales = 0
        for i in range(20):
            cact = buy_object(self.state)
            if cact == 1:
                profit_agent += self.state[0] - self.production_price
                agent_sales += 1
            elif cact == 2:
                self.comp_profit += self.state[2] - self.production_price
                comp_sales += 1

        # print("You sold " + str(agent_sales) +
        #       " and your competitor " + str(comp_sales))
        return self.state, profit_agent, self.counter >= 50, {}


# env = SimMarket()
# our_profit = 0
# is_done = False
# state = env.reset()
# print("The production price is " + str(env.production_price))
# while not is_done:
#     print("This is our state: " + str(state))
#     action = int(input("What do you want to do? "))
#     state, reward, is_done, _ = env.step(action)
#     print("Your profit this round is " + str(reward))
#     our_profit += reward
# print("In total you earned " + str(our_profit) +
#       " and your competitor " + str(env.comp_profit))
