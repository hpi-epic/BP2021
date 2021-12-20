#!/usr/bin/env python3

from abc import ABC, abstractmethod

import numpy as np

# helpers
import utils as ut


class Owner(ABC):
	"""
	The abstract class represents the owner of a product and is responsible for the decision what to do with the product.
	"""

	@abstractmethod
	def consider_return(self, others):
		raise NotImplementedError


class OwnerReturn(Owner):
	"""
	The class represents the owner of a product and is responsible for the decision of whether to return the product or not.
	"""

	def consider_return(self) -> None:
		"""
		The function returns the rebuy decision based on the probabilities calculated in the function set_probabilities_from_offer.
		The set_probabilities_from_offer function is not needed for this owner to function properly.
		"""
		return int(np.floor(np.random.rand() * 2))


class OwnerRebuy(OwnerReturn):
	"""
		In contrast to its superclass, this owner sells the product rather than returning it out of good will.
		For this it calculates a probability distribution of what to do with the product with every new offer and smaples from it.
	"""
	def __init__(self) -> None:
		super().__init__()
		self.probabilities = None

	def set_probabilities_from_offer(self, offer) -> None:
		"""
		The function calculates the probability distribution of a rebuy depending on the offer given by the buyer.

		Args:
			offer (np.array(int)): The offer given by the vendor.
		"""
		price_refurbished = offer[2] + 1
		price_new = offer[3] + 1
		price_rebuy = offer[4] + 1

		holding_preference = 1

		# If the price is low, the customer will discard the product
		discard_preference = 2 / (price_rebuy + 1)

		# Customer is very excited if the value of his product is close to the new or refurbished price
		return_preference = 2 * np.exp((price_rebuy - min(price_refurbished, price_new)) / min(price_refurbished, price_new))

		self.probabilities = ut.softmax(np.array([holding_preference, discard_preference, return_preference]))

	def consider_return(self) -> None:
		"""
		The function returns the rebuy decision based on the probabilities calculated in the function set_probabilities_from_offer. It has to be called first.
		"""
		assert self.probabilities is not None, 'Probabilities not set. You need to call set_probabilities_from_offer first.'
		return ut.shuffle_from_probabilities(self.probabilities)
