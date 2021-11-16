#!/usr/bin/env python3

# helper
import math
import random
import utils as ut
import numpy as np


# The following methods should be library calls in the future.
def softmax(preferences):
    exp_preferences = np.exp(preferences)
    return exp_preferences / sum(exp_preferences)


def shuffle_from_probabilities(probabilities):
    random_number = random.random()
    sum = 0
    for i, p in enumerate(probabilities):
        sum += p
        if random_number <= sum:
            return i
    return len(probabilities) - 1


class CustomerLinear:

    def buy_object(self, offers, nothingspreference=1): 
        random_val = random.random()
        abs_percent_price = (
            ut.PERCENTAGE_OF_RANDOM_CUSTOMERS + ut.PERCENTAGE_OF_PRICE_BASED_CUSTOMERS
        )
        if random_val < ut.PERCENTAGE_OF_RANDOM_CUSTOMERS:
            return self.buy_random(offers)
        elif random_val < abs_percent_price:
            return self.buy_price_based(offers)
        elif random_val < abs_percent_price + ut.PERCENTAGE_OF_QUALITY_BASED_CUSTOMERS:
            return self.buy_quality_based(offers)
        else:
            return self.buy_price_and_quality_based(offers, nothingspreference)

    def buy_random(self, offers):
        return random.randint(0, len(offers) / 2 - 1)

    def buy_price_and_quality_based(self, offers, nothingpreference):
        ratios = [nothingpreference]
        for i in range(int(len(offers) / 2)):
            ratio = offers[2 * i + 1] / offers[2 * i] - math.exp(offers[2 * i] - ut.MAX_PRICE - 3) 
            ratios.append(ratio)
        probabilities = softmax(np.array(ratios))
        return shuffle_from_probabilities(probabilities)

    def buy_price_based(self, offers):
        min_price_index = 0
        for i in range(int(len(offers) / 2)):
            if offers[2 * i] < offers[2 * min_price_index]:
                min_price_index = i
        return min_price_index

    def buy_quality_based(self, offers):
        max_quality_index = 0
        for i in range(int(len(offers) / 2)):
            if offers[2 * i + 1] < offers[2 * max_quality_index + 1]:
                max_quality_index = i
        return max_quality_index
