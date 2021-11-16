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

        if value_agent == 0 and value_compet == 0:
            return 0  # Don't buy anything
        elif value_agent > value_compet:
            return 1  # Buy agent's
        else:
            return 2  # Buy competitor's


class CustomerLinear(Customer):
    def buy_object(self, offers, nothingpreference=1):
        ratios = [nothingpreference]
        for i in range(int(len(offers) / 2)):
            ratio = offers[2 * i + 1] / offers[2 * i] - math.exp(offers[2 * i] - 27)
            ratios.append(ratio)
        probabilities = softmax(np.array(ratios))
        return shuffle_from_probabilities(probabilities)


class CustomerCircular(Customer):
    def buy_object(self, offers):
        ratio_new = 1 / offers[0] - math.exp(offers[0] - 8)
        ratio_old = 0.55 / offers[1] - math.exp(offers[1] - 5)
        return shuffle_from_probabilities(softmax(np.array([1, ratio_new, ratio_old])))
