import numpy as np

import recommerce.configuration.utils as ut
from recommerce.market.customer import Customer


class CustomerCircular(Customer):
	def generate_purchase_probabilities_from_offer(self, step_counter, common_state, vendor_specific_state, vendor_actions) -> np.array:
		"""
		This method calculates the purchase probability for each vendor in a linear setup.
		It is assumed that all vendors do have the same quality and same reputation.
		The customer values a second-hand-product 55% compared to a new one.

		Check the docstring in the superclass for interface description.
		"""
		assert isinstance(common_state, np.ndarray), 'common_state must be a np.ndarray'
		assert isinstance(vendor_specific_state, list), 'vendor_specific_state must be a list'
		assert isinstance(vendor_actions, list), 'vendor_actions must be a list'
		assert len(vendor_specific_state) == len(vendor_actions), \
			'Both the vendor_specific_state and vendor_actions contain one element per vendor. So they must have the same length.'
		assert len(vendor_specific_state) > 0, 'there must be at least one vendor.'

		nothingpreference = 1
		preferences = [nothingpreference]
		for vendor_idx in range(len(vendor_actions)):
			price_refurbished = vendor_actions[vendor_idx][0] + 1
			price_new = vendor_actions[vendor_idx][1] + 1
			assert price_refurbished >= 1 and price_new >= 1, 'price_refurbished and price_new need to be >= 1'

			ratio_old = 5.5 / price_refurbished - np.exp(price_refurbished - 5)
			ratio_new = 10 / price_new - np.exp(price_new - 8)
			preferences += [ratio_old, ratio_new]

		return ut.softmax(np.array(preferences))
