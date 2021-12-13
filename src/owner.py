#!/usr/bin/env python3

from abc import ABC, abstractmethod

import numpy as np

# helpers
import utils as ut


class Owner(ABC):
	"""
	The abstract class represents the owner of a product and is responsible for the decision what to do with the product.
	"""

	def __init__(self) -> None:
		pass

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
		"""
		return int(np.floor(np.random.rand() * 2))


class OwnerRebuy(OwnerReturn):
	"""
	An owner return represents a person who owns a product and might wants to sell it back to the store.In contrast to a return, a rebuy is a person who owns a product and wants to sell it to the vendor rather than to return it for free.
	"""
	def __init__(self) -> None:
		super().__init__()
		self.probabilities = None

	def set_probabilities_from_offer(self, offer):
		"""
		The function calculates the probabilities of a rebuy depending on the offer given by the buyer.

		Args:
			offer (np.array(int)): The offer given by the vendor.
		"""
		holding_preference = 1

		# If the price is low, the customer will discard the product
		discard_preference = 2 / (offer[2] + 1)

		# Customer is very excited if the value of his product is close to the new or refurbished price
		return_preference = 2 * np.exp((offer[2] - min(offer[0], offer[1])) / min(offer[0], offer[1]))

		self.probabilities = ut.softmax(np.array([holding_preference, discard_preference, return_preference]))

	def consider_return(self) -> None:
		"""
		The function returns the rebuy decision based on the probabilities calculated in the function set_probabilities_from_offer.
		"""
		assert self.probabilities is not None, 'Probabilities not set. You need to call set_probabilities_from_offer first.'
		return ut.shuffle_from_probabilities(self.probabilities)
