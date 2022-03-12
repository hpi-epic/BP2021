import numpy as np
import pytest

import market.circular.circular_sim_market as circular_market
import market.customer as customer
import market.linear.linear_sim_market as linear_market


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour.
# This is dependent on the sim_market working!
def random_offer(marketplace):
	marketplace = marketplace()
	marketplace.reset()
	marketplace.vendor_actions[0] = marketplace._action_space.sample()
	return marketplace._get_common_state_array(), marketplace.vendor_specific_state, marketplace.vendor_actions


# Test the Customer parent class, i.e. make sure it cannot be used
def test_customer_parent_class():
	with pytest.raises(NotImplementedError) as assertion_message:
		customer.Customer.generate_purchase_probabilities_from_offer(customer.CustomerLinear, *random_offer(linear_market.ClassicScenario))
	assert 'This method is abstract. Use a subclass' in str(assertion_message.value)


# the following list contains invalid parameters for generate_purchase_probabilities_from_offer and the expected error messages
generate_purchase_probabilities_from_offer_testcases = [
	(customer.CustomerLinear(), [20, 20], 1, 'offer_length_per_vendor must be two: one field for the price and one for the quality!'),
	(customer.CustomerCircular(), [20, 20], 1, 'offers must be a np.ndarray'),
	(customer.CustomerCircular(), np.array([20, 20, 20, 20]), 4, 'there must be exactly one field for common state (in_circulation)'),
	(customer.CustomerCircular(), np.array([20, 20, 20, 20, 20, 20]), 5, 'offer_length_per_vendor needs to be 3 or 4'),
	(customer.CustomerCircular(), np.array([-20, -20, -20, -20]), 3, 'price_refurbished and price_new need to be >= 1')
]


@pytest.mark.parametrize(
	'customer, offers, offer_length_per_vendor, expected_message',
	generate_purchase_probabilities_from_offer_testcases)
def test_generate_purchase_probabilities_from_offer(customer, offers, offer_length_per_vendor, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		customer.generate_purchase_probabilities_from_offer(offers, offer_length_per_vendor)
	assert expected_message in str(assertion_message.value)


customer_action_range_testcases = [
	(customer.CustomerLinear, linear_market.ClassicScenario),
	(customer.CustomerLinear, linear_market.MultiCompetitorScenario),
	(customer.CustomerCircular, circular_market.CircularEconomyMonopolyScenario),
	(customer.CustomerCircular, circular_market.CircularEconomyRebuyPriceMonopolyScenario)
]


# Test the different Customers in the different Market Scenarios
@pytest.mark.parametrize('customer, market', customer_action_range_testcases)
def test_customer_action_range(customer, market):
	offers = random_offer(market)
	probability_distribution = customer.generate_purchase_probabilities_from_offer(customer, *offers)
	assert len(probability_distribution) == market()._get_number_of_vendors() * \
		(1 if issubclass(market, linear_market.LinearEconomy) else 2) + 1
