#!/usr/bin/env python3

# helper
import math
import random

import numpy as np

import utils as ut


class Competitor:
    def __init__(self):
        self.maxprice = ut.MAX_PRICE
        self.maxquality = ut.MAX_QUALITY
        self.quality = ut.shuffle_quality()

    def get_initial_price(self, random_start):
        if random_start:
            return max(
                int(ut.PRODUCTION_PRICE + np.random.normal() * 3 + 3),
                ut.PRODUCTION_PRICE + 1,
            )
        else:
            return ut.PRODUCTION_PRICE

    def reset(self, random_start):
        self.quality = ut.shuffle_quality()
        return self.get_initial_price(random_start), self.quality

    def give_competitors_price(self, state, self_idx):
        assert False, 'You must use a subclass of Competitor!'


class CompetitorLinearRatio1(Competitor):
    def give_competitors_price(self, state, self_idx):
        # this stratgy is based on a price quality ratio
        ratios = []
        max_competing_ratio = 0
        for i in range(math.floor(len(state) / 2)):
            ratio = state[2 * i + 1] / state[2 * i]
            ratios.append(ratio)
            if ratio > max_competing_ratio and i != self_idx:
                max_competing_ratio = ratio

        ratio = max_competing_ratio / ratios[self_idx]
        randomnumber = random.random()
        intended = 0
        if randomnumber < 0.15:
            intended = state[2 * self_idx] + random.randint(-1, 1)
        elif randomnumber < 0.3:
            intended = random.randint(ut.PRODUCTION_PRICE, ut.MAX_PRICE)
        else:
            intended = math.floor(max_competing_ratio * state[2 * self_idx + 1])
        return min(max(1, intended), ut.MAX_PRICE)


class CompetitorRandom(Competitor):
    def give_competitors_price(self, _):
        return random.randint(ut.PRODUCTION_PRICE, ut.MAX_PRICE)


class CompetitorJust2Players(Competitor):
    def give_competitors_price(self, state, _):
        # this competitor is based on quality and agents actions
        assert len(state) == 4, "You can't use this competitor in this market!"
        agent_price = state[0]
        comp_price = state[2]
        agent_quality = state[1]
        comp_quality = state[3]

        new_price = 0

        if comp_quality > agent_quality + 15:
            # significantly better quality
            new_price = agent_price + 2
        elif comp_quality > agent_quality:
            # slightly better quality
            new_price = agent_price + 1
        elif comp_quality < agent_quality and comp_quality > agent_quality - 15:
            # slightly worse quality
            new_price = agent_price - 1
        elif comp_quality < agent_quality:
            # significantly worse quality
            new_price = agent_price - 2
        elif comp_quality == agent_quality:
            # same quality
            new_price = comp_price
        if new_price < ut.PRODUCTION_PRICE:
            new_price = ut.PRODUCTION_PRICE + 1
        elif new_price > self.maxprice:
            new_price = self.maxprice
        return new_price
