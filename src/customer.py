#!/usr/bin/env python3

# helpers
import math
from typing import Tuple

import numpy as np

import utils as ut

# import random


class Customer:
	def __init__(self) -> None:
		self.probabilities = None

	def buy_object(self, others):
		assert False, 'This class should not be used.'


# This customer is only useful in a two-players setup. We consider to replace it fully
# class CustomerDeprecated(Customer):
# 	def buy_object(self, offers):
# 		if random.random() < 0.17:
# 			return random.randint(1, 2)
# 		value_agent = max(offers[1] / offers[0] + np.random.normal() / 2, 0.1)
# 		value_compet = max(offers[3] / offers[2] + np.random.normal() / 2, 0.1)
# 		maxprice = np.random.normal() * 3 + 25
# 		if offers[0] > maxprice:
# 			value_agent = 0
# 		if offers[2] > maxprice:
# 			value_compet = 0

# 		customer_buy = 0
# 		if value_agent == 0 and value_compet == 0:
# 			customer_buy = 0  # Don't buy anything
# 		elif value_agent > value_compet:
# 			customer_buy = 1  # Buy agent's
# 		else:
# 			customer_buy = 2  # Buy competitor's
# 		return customer_buy, None


class CustomerLinear(Customer):
	def __init__(self) -> None:
		super().__init__()

	def set_probabilities_from_offers(self, offers, nothingpreference=1) -> None:
		ratios = [nothingpreference]
		for i in range(int(len(offers) / 2)):
			ratio = offers[2 * i + 1] / (offers[2 * i] + 1) - math.exp(offers[2 * i] - 27)
			ratios.append(ratio)
		self.probabilities = ut.softmax(np.array(ratios))

	# This customer calculates the value per money for each vendor and chooses those with high value with a higher probability
	def buy_object(self, offers, nothingpreference=1) -> int:
		return ut.shuffle_from_probabilities(self.probabilities)


class CustomerCircular(Customer):
	def __init__(self) -> None:
		super().__init__()

	# This customer values a second-hand-product 55% of a new product
	def set_probabilities_from_offers(self, offers) -> None:
		price_refurbished = offers[2]
		price_new = offers[3]
		assert price_refurbished >= 1 and price_new >= 1

		ratio_old = 5.5 / price_refurbished - math.exp(price_refurbished - 5)
		ratio_new = 10 / price_new - math.exp(price_new - 8)
		preferences = np.array([1, ratio_old, ratio_new])
		self.probabilities = ut.softmax(preferences)

	def buy_object(self, offers) -> Tuple[int, int]:
		customer_desicion = ut.shuffle_from_probabilities(self.probabilities)
		return customer_desicion
