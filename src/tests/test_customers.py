import numpy as np
import pytest

import configuration.utils_sim_market as ut
import market.customer as customer
from market.customer import CustomerCircular as CCircular
from market.customer import CustomerLinear as CLinear
from market.sim_market import CircularEconomyMonopolyScenario as CEMonopoly
from market.sim_market import CircularEconomyRebuyPriceMonopolyScenario
from market.sim_market import ClassicScenario as SClassic
from market.sim_market import MultiCompetitorScenario as SMulti


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer(market_scenario):
	market = market_scenario()
	market.reset()
	market.vendor_actions[0] = market.action_space.sample()
	return market.generate_customer_offer(), market.get_offer_length_per_vendor()


# Test the Customer parent class, i.e. make sure it cannot be used
def test_customer_parent_class():
	with pytest.raises(AssertionError) as assertion_info:
		customer.Customer.generate_purchase_probabilities_from_offer(CLinear, *random_offer(SClassic))
	assert str(assertion_info.value) == 'This class should not be used.'


# the following list contains invalid parameters for generate_purchase_probabilities_from_offer and the expected error messages
invalid_values = [
	(CLinear(), [20, 20], 1, 'offer_length_per_vendor must be two: one field for the price and one for the quality!'),
	(CCircular(), [20, 20], 1, 'offers must be a np.array'),
	(CCircular(), np.array([20, 20, 20, 20]), 4, 'there must be exactly one field for common state (in_circulation)'),
	(CCircular(), np.array([20, 20, 20, 20, 20, 20]), 5, 'offer_length_per_vendor needs to be  3 or 4'),
	(CCircular(), np.array([-20, -20, -20, -20]), 3, 'price_old and price_new need to be greater or equal 1')
]


@pytest.mark.parametrize('customer, offers, offer_length_per_vendor, expected_error_msg', invalid_values)
def test_generate_purchase_probabilities_from_offer_assertions(customer, offers, offer_length_per_vendor, expected_error_msg):
	with pytest.raises(AssertionError) as assertion_info:
		customer.generate_purchase_probabilities_from_offer(offers, offer_length_per_vendor)
	assert str(assertion_info.value) == expected_error_msg


array_customer_action_range = [
	(CLinear, *random_offer(SClassic), 4), (CLinear, *random_offer(SMulti), 8), (CCircular, *random_offer(CEMonopoly), 4), (CCircular, *random_offer(CircularEconomyRebuyPriceMonopolyScenario), 5)
]


# mark.parametrize can be used to run the same test with different parameters
# Test the different Customers in the different Market Scenarios
@pytest.mark.parametrize('customer, offers, offer_length_per_vendor, expectedSize', array_customer_action_range)
def test_customer_action_range(customer, offers, offer_length_per_vendor, expectedSize):
	assert len(offers) == expectedSize
	probability_distribution = customer.generate_purchase_probabilities_from_offer(customer, offers, offer_length_per_vendor)
	buy_decisions = ut.shuffle_from_probabilities(probability_distribution)
	assert 0 <= buy_decisions <= expectedSize - 1
