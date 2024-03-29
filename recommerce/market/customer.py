from abc import ABC, abstractmethod

import numpy as np


class Customer(ABC):
	@abstractmethod
	def generate_purchase_probabilities_from_offer(self, common_state, vendor_specific_state, vendor_actions) -> np.array:  # pragma: no cover
		"""
		This method receives the state of the market and uses it as a list of offers.
		It returns the purchase probability for all vendors.
		Args:
			common_state (np.array): The common state array generated by the market
			vendor_specific_state (list): The array of arrays whith an entry for each vendor
			vendor_actions (list): The array containing the action for each vendor
		Returns:
			np.array: probability distribution for all possible purchase decisions.
			In the first field, there is the probability that the customer does not buy anything.
			In the subsequent fields, there are the probabilites for buying the specific offers from the vendor.
			Look subclass implementation for more details.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')
