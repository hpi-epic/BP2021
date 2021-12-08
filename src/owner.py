#!/usr/bin/env python3

# helpers
import numpy as np

import utils as ut

# import random


class Owner:
	def __init__(self) -> None:
		pass

	def return_object(self, others):
		assert False, 'This class should not be used.'


class OwnerReturn(Owner):
	def __init__(self) -> None:
		super().__init__()

	def consider_return(self, offer, profits) -> None:
		return int(np.floor(np.random.rand() * 2))


class OwnerRebuy(OwnerReturn):
	def __init__(self) -> None:
		super().__init__()
		self.probabilities = None

	def set_probabilities_from_offer(self, offer):
		holding_preference = 1

		# If the price is low, the customer will discard the product
		discard_preference = 2 / (offer[2] + 1)

		# Customer is very excited if the value of his product is close to the new or refurbished price
		return_preference = 2 * np.exp((offer[2] - min(offer[0], offer[1])) / min(offer[0], offer[1]))

		self.probabilities = ut.softmax(np.array([holding_preference, discard_preference, return_preference]))

	def consider_return(self, offer, profits) -> None:
		assert self.probabilities is not None, 'Probabilities not set. You need to call set_probabilities_from_offer first.'
		return ut.shuffle_from_probabilities(self.probabilities)
