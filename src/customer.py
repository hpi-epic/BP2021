#!/usr/bin/env python3

# helper
import math
import random

import numpy as np


# The following methods should be library calls in the future.
def softmax(preferences):
    exp_preferences = np.exp(preferences)
    return exp_preferences / sum(exp_preferences)


def shuffle_from_probabilities(probabilities):
    randomnumber = random.random()
    sum = 0
    for i, p in enumerate(probabilities):
        sum += p
        if randomnumber <= sum:
            return i
    return len(probabilities) - 1


class Customer:
    def buy_object(self, others):
        assert False, 'This class should not be used.'


# This customer is only useful in a two-players setup. We consider to replace it fully
class CustomerDeprecated(Customer):
    def buy_object(self, offers):
        if random.random() < 0.17:
            return random.randint(1, 2)
        value_agent = max(offers[1] / offers[0] + np.random.normal() / 2, 0.1)
        value_compet = max(offers[3] / offers[2] + np.random.normal() / 2, 0.1)
        maxprice = np.random.normal() * 3 + 25
        if offers[0] > maxprice:
            value_agent = 0
        if offers[2] > maxprice:
            value_compet = 0

        customer_buy = 0
        if value_agent == 0 and value_compet == 0:
            customer_buy = 0  # Don't buy anything
        elif value_agent > value_compet:
            customer_buy = 1  # Buy agent's
        else:
            customer_buy = 2  # Buy competitor's
        return customer_buy, None


class CustomerLinear(Customer):
    # This customer calculates the value per money for each vendor and chooses those with high value with a higher probability
    def buy_object(self, offers, nothingpreference=1):
        ratios = [nothingpreference]
        for i in range(int(len(offers) / 2)):
            ratio = offers[2 * i + 1] / offers[2 * i] - math.exp(offers[2 * i] - 27)
            ratios.append(ratio)
        probabilities = softmax(np.array(ratios))
        return shuffle_from_probabilities(probabilities), None


class CustomerCircular(Customer):
    # This customer values a second-hand-product 55% of a new product
    def buy_object(self, offers):
        assert offers[0] >= 1 and offers[1] >= 1
        ratio_old = 5.5 / offers[0] - math.exp(offers[0] - 5)
        ratio_new = 10 / offers[1] - math.exp(offers[1] - 8)
        preferences = np.array([1, ratio_old, ratio_new])
        probabilities = softmax(preferences)
        return shuffle_from_probabilities(probabilities), 1 if np.random.rand() < 0.05 * offers[3] / 20 else None
