from abc import ABC, abstractmethod

import numpy as np

import recommerce.configuration.utils as ut


class Owner(ABC):
	"""
	The abstract class represents the owner of a product and is responsible for the decision of whether to return the product or not.
	"""

	@abstractmethod
	def generate_return_probabilities_from_offer(self, common_state, vendor_specific_state, vendor_actions) -> np.array:
		"""
		This abstract method receives offers and generates the probabilities for the possible owner actions.
		An owner can throw away his product, he can hold his product or return it to one of the vendors
		Args:
			common_state (np.array): The common state array generated by the market
			vendor_specific_state (list): The array of arrays whith an entry for each vendor
			vendor_actions (list): The array containing the action for each vendor
		Returns:
			np.array: The first entry is the probability that the holds his product.
			The second entry is the probability that the owner throws away his product.
			Afterwards, for all vendors the probabilities of returning the product to him follows.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')


class UniformDistributionOwner(Owner):
	def generate_return_probabilities_from_offer(self, common_state, vendor_specific_state, vendor_actions) -> np.array:
		"""
		This method generates a uniform distribution over all vendors
		to return the product without considering the rebuy price and the overall market situation.
		It assumes three entries per vendor (refurbished price, new price and in_storage but NO rebuy price)
		Check the docstring in the superclass for interface description.
		"""
		assert isinstance(common_state, np.ndarray), 'offers needs to be a np.ndarray'
		assert isinstance(vendor_specific_state, list), 'vendor_specific_state must be a list'
		assert isinstance(vendor_actions, list), 'vendor_actions must be a list'
		assert len(vendor_specific_state) == len(vendor_actions), \
			'Both the vendor_specific_state and vendor_actions contain one element per vendor. So they must have the same length.'
		assert len(vendor_specific_state) > 0, 'there must be at least one vendor.'

		number_of_options = len(vendor_specific_state) + 2
		return np.array([1 / number_of_options] * int(number_of_options))


class OwnerRebuy(Owner):
	def generate_return_probabilities_from_offer_old(self, common_state, vendor_specific_state, vendor_actions) -> np.array:
		"""
		This method tries a more sophisticated version of generating return probabilities.
		The owner likes if the rebuy price is close to the price in the sell offer.
		That will increase the probability that he will return his product.
		If the rebuy price is very low, the owner will just throw away his product more often. Holding the product is the fallback option.
		Check the docstring in the superclass for interface description.
		"""
		assert isinstance(common_state, np.ndarray), 'offers needs to be a ndarray'
		assert isinstance(vendor_specific_state, list), 'vendor_specific_state must be a list'
		assert isinstance(vendor_actions, list), 'vendor_actions must be a list'
		assert len(vendor_specific_state) == len(vendor_actions), \
			'Both the vendor_specific_state and vendor_actions contain one element per vendor. So they must have the same length.'
		assert len(vendor_specific_state) > 0, 'there must be at least one vendor.'

		holding_preference = 1
		discard_preference = 20
		return_preferences = []

		for vendor_idx in range(len(vendor_specific_state)):
			price_refurbished = vendor_actions[vendor_idx][0] + 1
			price_new = vendor_actions[vendor_idx][1] + 1
			price_rebuy = vendor_actions[vendor_idx][2] + 1
			best_purchase_offer = min(price_refurbished, price_new)
			return_preferences.append(2 * np.exp((price_rebuy - best_purchase_offer) / best_purchase_offer))

			discard_preference = min(discard_preference, 2 / (price_rebuy + 1))

		return ut.softmax(np.array([holding_preference, discard_preference] + return_preferences))

	def generate_return_probabilities_from_offer(self, common_state, vendor_specific_state, vendor_actions) -> np.array:
		"""
		This method tries a more sophisticated version of generating return probabilities.
		The owner likes if the rebuy price is close to the price in the sell offer.
		That will increase the probability that he will return his product.
		If the rebuy price is very low, the owner will just throw away his product more often. Holding the product is the fallback option.
		Check the docstring in the superclass for interface description.
		"""
		assert isinstance(common_state, np.ndarray), 'offers needs to be a ndarray'
		assert isinstance(vendor_specific_state, list), 'vendor_specific_state must be a list'
		assert isinstance(vendor_actions, list), 'vendor_actions must be a list'
		assert len(vendor_specific_state) == len(vendor_actions), \
			'Both the vendor_specific_state and vendor_actions contain one element per vendor. So they must have the same length.'
		assert len(vendor_specific_state) > 0, 'there must be at least one vendor.'

		holding_preference = 1
		return_preferences = []
		lowest_purchase_offer = 100000
		best_rebuy_price = 0

		for vendor_idx in range(len(vendor_specific_state)):
			price_refurbished = vendor_actions[vendor_idx][0] + 1
			price_new = vendor_actions[vendor_idx][1] + 1
			price_rebuy = vendor_actions[vendor_idx][2] + 1
			best_rebuy_price = max(best_rebuy_price, price_rebuy)
			lowest_purchase_offer = min(lowest_purchase_offer, price_refurbished, price_new)
			return_preferences.append(price_rebuy)

		discard_preference = lowest_purchase_offer - best_rebuy_price

		return ut.softmax(np.array([holding_preference, discard_preference] + return_preferences))
