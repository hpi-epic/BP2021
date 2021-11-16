#!/usr/bin/env python3

# helper
import math
import random

import utils as ut


class Competitor:
    def __init__(self):
        self.maxprice = ut.MAX_PRICE
        self.maxquality = ut.MAX_QUALITY
        self.quality = ut.shuffle_quality()

    def reset(self):
        return (
            # 1 will be added to the price because profit is unpossible else
            random.randint(ut.PRODUCTION_PRICE + 1, ut.MAX_PRICE),
            ut.shuffle_quality(),
        )

    def give_competitors_price(self, state, self_idx):
        assert False, 'You must use a subclass of Competitor!'


class CompetitorLinearRatio1(Competitor):
    def give_competitors_price(self, state, self_idx):
        # this stratgy calculates the value per money for each competing vendor and tries to adept to it
        ratios = []
        max_competing_ratio = 0
        for i in range(math.floor(len(state) / 2)):
            ratio = state[2 * i + 1] / state[2 * i]  # value for money for vendor i
            ratios.append(ratio)
            if ratio > max_competing_ratio and i != self_idx:
                max_competing_ratio = ratio

        ratio = max_competing_ratio / ratios[self_idx]
        randomnumber = random.random()
        intended = 0
        if randomnumber < 0.15:
            intended = state[2 * self_idx] + random.randint(-1, 1)
        else:
            intended = math.floor(1 / max_competing_ratio * state[2 * self_idx + 1])
        return min(min(max(ut.PRODUCTION_PRICE + 1, intended), ut.MAX_PRICE), 23)


class CompetitorRandom(Competitor):
    def give_competitors_price(self, state, _):
        return random.randint(ut.PRODUCTION_PRICE + 1, ut.MAX_PRICE)


class CompetitorBasedOnAgent(Competitor):
    def give_competitors_price(self, state, _):
        # This competitor is based on quality and agents actions.
        # While he can act in every linear economy, you should not expect good performance in a multicompetitor setting.
        # assert len(state) == 4, "You can't use this competitor in this market!"
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
