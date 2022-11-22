import numpy as np
import math
import recommerce.configuration.utils as ut
from recommerce.market.customer import Customer


class CustomerLinear(Customer):
	def generate_purchase_probabilities_from_offer(self, common_state, vendor_specific_state, vendor_actions) -> np.array:
		"""
		This method calculates the purchase probability for each vendor in a linear setup.
		Quality values are used to calculate a ratio.
		Customers will follow that ratio.

		Check the docstring in the superclass for interface description.
		"""
		assert isinstance(common_state, np.ndarray), 'common_state must be a np.ndarray'
		# assert len(common_state) == 1, 'common_state must be an empty array in our linear setup'
		assert isinstance(vendor_specific_state, list), 'vendor_specific_state must be a list'
		assert isinstance(vendor_actions, list), 'vendor_actions must be a list'
		assert len(vendor_specific_state) == len(vendor_actions), \
			'Both the vendor_specific_state and vendor_actions contain one element per vendor. So they must have the same length.'
		assert len(vendor_specific_state) > 0, 'there must be at least one vendor.'

		# hacky ...
		for idx, element in enumerate(vendor_actions):
			if len(element.shape) == 1:
				vendor_actions[idx] = np.array(vendor_actions[idx][0])

		step = common_state[1]
		seasonal_interest = 7 * (math.sin(step * (math.pi / 2) * (1 / 50)))

		# seasonal_interest = 0


		nothingpreference = 1
		ratios = [nothingpreference]
		for vendor_idx in range(len(vendor_actions)):
			quality = vendor_specific_state[vendor_idx][0]
			price = vendor_actions[vendor_idx] + 1
			# ratio = quality / price
			ratio = 10 / price - np.exp(price - 8)
			ratio += seasonal_interest
			ratios.append(ratio)
		return ut.softmax(np.array(ratios))
