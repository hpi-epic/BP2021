import math

import numpy as np

import recommerce.configuration.utils as ut
import scipy.stats
from recommerce.market.customer import Customer


class CustomerLinear(Customer):
    def generate_purchase_probabilities_from_offer(self, step_counter, common_state, vendor_specific_state,
                                                   vendor_actions) -> np.array:
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


        MAX_PRICE = 10 # introduce config
        mu = 4
        low_demand_reference_price = 0.5 * MAX_PRICE
        high_demand_reference_price = 0.9 * MAX_PRICE

        normal = scipy.stats.norm(50, 6)
        current_demand = normal.pdf(step_counter % 100) / normal.pdf(50)
        x = [0, 1]
        y = [low_demand_reference_price, high_demand_reference_price]
        reference_price = np.interp(current_demand, x, y)

        nothing_preference = 1
        ratios = [nothing_preference]
        for vendor_idx in range(len(vendor_actions)):
            price = vendor_actions[vendor_idx]
            ratio = mu * (-np.exp(price-reference_price) + reference_price) / reference_price
            ratios.append(ratio)
        return ut.softmax(np.array(ratios))
