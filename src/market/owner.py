#!/usr/bin/env python3

from abc import ABC, abstractmethod

import numpy as np

# helpers
import configuration.utils_sim_market as ut


class Owner(ABC):
	"""
	The abstract class represents the owner of a product and is responsible for the decision of whether to return the product or not.
	"""

	@abstractmethod
	def generate_return_probabilities_from_offer(self, offers, offer_length_per_vendor) -> np.array:
		"""This abstract method receives offers and generates the probabilities for the possible owner actions. An owner can throw away his product, he can hold his product or return it to one of the vendors

		Args:
			offers (np.array): a rebuy offer an owner receives

		Returns:
			np.array: The first entry is the probability that the holds his product. The second entry is the probability that the owner throws away his product. Afterwards, for all vendors the probabilities of returning the product to him follows.
		"""
		raise NotImplementedError


class UniformDistributionOwner(Owner):
	def generate_return_probabilities_from_offer(self, offers, offer_length_per_vendor) -> np.array:
		"""This method generates a uniform distribution over all vendors to return the product without considering the rebuy price and the overall market situation.
		It assumes three entries per vendor (refurbished price, new price and in_storage but NO rebuy price)

		Args:
			offers (np.array): The content is ignored in this method. It is just used to determine the number of options which is the number of vendors plus 2 (throw away, hold the product).

		Returns:
			np.array: a uniform distribution over all possible actions
		"""
		assert isinstance(offers, np.ndarray), 'offers needs to be an ndarray'
		assert len(offers) % offer_length_per_vendor == 1, 'one of offer_length_per_vendor and len(offers) must be even, the other odd'
		number_of_options = np.floor(len(offers) / 3) + 2
		return np.array([1 / number_of_options] * int(number_of_options))


class OwnerRebuy(Owner):
	def generate_return_probabilities_from_offer(self, offers, offer_length_per_vendor) -> np.array:
		"""This method tries a more sophisticated version of generating return probabilities.
		The owner likes if the rebuy price is close to the price in the sell offer.
		That will increase the probability that he will return his product.
		If the rebuy price is very low, the owner will just throw away his product more often. Holding the product is the fallback option.

		Args:
			offers (np.array): The method assumes one entry for the number of products in circulation and four entries for each vendor (refurbished price, new price, rebuy price and in_storage)

		Returns:
			np.array: probability distribution over all possible actions.
		"""
		assert isinstance(offers, np.ndarray), 'offers needs to be an ndarray'
		assert len(offers) % offer_length_per_vendor == 1, 'one of offer_length_per_vendor and len(offers) must be even, the other odd'

		holding_preference = 1
		discard_preference = 20
		return_preferences = []

		for offer in range(int(np.floor(len(offers) / offer_length_per_vendor))):
			price_refurbished = offers[1 + offer * offer_length_per_vendor] + 1
			price_new = offers[2 + offer * offer_length_per_vendor] + 1
			price_rebuy = offers[3 + offer * offer_length_per_vendor] + 1
			best_purchase_offer = min(price_refurbished, price_new)
			return_preferences.append(2 * np.exp((price_rebuy - best_purchase_offer) / best_purchase_offer))

			discard_preference = min(discard_preference, 2 / (price_rebuy + 1))

		return ut.softmax(np.array([holding_preference, discard_preference] + return_preferences))
