#!/usr/bin/env python3

# helper
import random
import numpy as np

class Customer(): 
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