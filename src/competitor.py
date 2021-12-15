#!/usr/bin/env python3

# helper
import math
import random

import utils as ut
from agent import Agent


class CompetitorLinearRatio1(Agent):
	def policy(self, state, epsilon=0):
		# print(state)
		# this stratgy calculates the value per money for each competing vendor and tries to adapt to it
		ratios = []
		# ratios[0] is the ratio of the competitor itself. it is compared to the other ratios
		max_competing_ratio = 0
		for i in range(math.floor(len(state) / 2)):
			ratio = state[2 * i + 2] / state[2 * i + 1]  # value for price for vendor i
			ratios.append(ratio)
			if ratio > max_competing_ratio:
				max_competing_ratio = ratio

		ratio = max_competing_ratio / ratios[0]
		intended = math.floor(1 / max_competing_ratio * state[0]) - 1
		actual_price = min(max(ut.PRODUCTION_PRICE + 1, intended), ut.MAX_PRICE - 1)
		# print('price from the competitor:', actual_price)
		return actual_price


class CompetitorRandom(Agent):
	def policy(self, state, epsilon=0):
		return random.randint(ut.PRODUCTION_PRICE + 1, ut.MAX_PRICE - 1)


class CompetitorJust2Players(Agent):
	def policy(self, state, epsilon=0):
		# This competitor is based on quality and agents actions.
		# While he can act in every linear economy, you should not expect good performance in a multicompetitor setting.
		# assert len(state) == 4, "You can't use this competitor in this market!"
		agent_price = state[1]
		agent_quality = state[2]
		comp_quality = state[0]

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
			new_price = agent_price
		if new_price < ut.PRODUCTION_PRICE:
			new_price = ut.PRODUCTION_PRICE + 1
		elif new_price > ut.MAX_PRICE:
			new_price = ut.MAX_PRICE
		new_price = int(new_price)
		assert isinstance(new_price, int)
		return new_price
