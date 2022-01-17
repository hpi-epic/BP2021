import numpy as np
import pytest

import configuration.utils as ut
import market.customer as customer
import market.sim_market as sim_market


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer(marketplace):
	marketplace = marketplace()
	marketplace.reset()
	marketplace.vendor_actions[0] = marketplace._action_space.sample()
	return marketplace._generate_customer_offer(), marketplace._get_offer_length_per_vendor()


# Test the Customer parent class, i.e. make sure it cannot be used
def test_customer_parent_class():
	with pytest.raises(NotImplementedError) as assertion_message:
		customer.Customer.generate_purchase_probabilities_from_offer(customer.CustomerLinear, random_offer(sim_market.ClassicScenario), 1)
	assert 'This method is abstract. Use a subclass' in str(assertion_message.value)


# the following list contains invalid parameters for generate_purchase_probabilities_from_offer and the expected error messages
generate_purchase_probabilities_from_offer_testcases = [
	(customer.CustomerLinear(), [20, 20], 1, 'offer_length_per_vendor must be two: one field for the price and one for the quality!'),
	(customer.CustomerCircular(), [20, 20], 1, 'offers must be a np.ndarray'),
	(customer.CustomerCircular(), np.array([20, 20, 20, 20]), 4, 'there must be exactly one field for common state (in_circulation)'),
	(customer.CustomerCircular(), np.array([20, 20, 20, 20, 20, 20]), 5, 'offer_length_per_vendor needs to be 3 or 4'),
	(customer.CustomerCircular(), np.array([-20, -20, -20, -20]), 3, 'price_refurbished and price_new need to be >= 1')
]


@pytest.mark.parametrize('customer, offers, offer_length_per_vendor, expected_message', generate_purchase_probabilities_from_offer_testcases)
def test_generate_purchase_probabilities_from_offer(customer, offers, offer_length_per_vendor, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		customer.generate_purchase_probabilities_from_offer(offers, offer_length_per_vendor)
	assert expected_message in str(assertion_message.value)


customer_action_range_testcases = [
	(customer.CustomerLinear, *random_offer(sim_market.ClassicScenario), 4),
	(customer.CustomerLinear, *random_offer(sim_market.MultiCompetitorScenario), 8),
	(customer.CustomerCircular, *random_offer(sim_market.CircularEconomyMonopolyScenario), 4),
	(customer.CustomerCircular, *random_offer(sim_market.CircularEconomyRebuyPriceMonopolyScenario), 5)
]


# Test the different Customers in the different Market Scenarios
@pytest.mark.parametrize('customer, offers, offer_length_per_vendor, expected_size', customer_action_range_testcases)
def test_customer_action_range(customer, offers, offer_length_per_vendor, expected_size):
	assert len(offers) == expected_size
	probability_distribution = customer.generate_purchase_probabilities_from_offer(customer, offers, offer_length_per_vendor)
	buy_decisions = ut.shuffle_from_probabilities(probability_distribution)
	assert 0 <= buy_decisions <= expected_size - 1
