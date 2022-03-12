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
	# (customer.CustomerLinear, np.array([]), [[12], [15]], [3, 5], 'this should work'),
	(customer.CustomerLinear, [], [[12], [15]], [3, 5], 'common_state must be a np.ndarray'),
	(customer.CustomerLinear, np.array([]), ([12], [15]), [3, 5], 'vendor_specific_state must be a list'),
	(customer.CustomerLinear, np.array([]), [[12], [15]], (3, 5), 'vendor_actions must be a list'),
	(customer.CustomerLinear, np.array([]), [[12]], [3, 5], 'they must have the same length'),
	(customer.CustomerLinear, np.array([]), [[12], [15]], [3], 'they must have the same length'),
	(customer.CustomerLinear, np.array([]), [], [], 'there must be at least one vendor'),
	# (customer.CustomerCircular, np.array([]), [[17], [23]], [[3, 6], [4, 7]], 'this should work'),
	(customer.CustomerCircular, [], [[17], [23]], [[3, 6], [4, 7]], 'common_state must be a np.ndarray'),
	(customer.CustomerLinear, np.array([]), ([17], [23]), [[3, 6], [4, 7]], 'vendor_specific_state must be a list'),
	(customer.CustomerCircular, np.array([]), [[17], [23]], ([3, 6], [4, 7]), 'vendor_actions must be a list'),
	(customer.CustomerCircular, np.array([]), [[17]], [[3, 6], [4, 7]], 'they must have the same length'),
	(customer.CustomerCircular, np.array([]), [[17], [23]], [[3, 6]], 'they must have the same length'),
	(customer.CustomerCircular, np.array([]), [], [], 'there must be at least one vendor'),
]


@pytest.mark.parametrize(
	'customer, common_state, vendor_specific_state, vendor_actions, expected_message',
	generate_purchase_probabilities_from_offer_testcases)
def test_generate_purchase_probabilities_from_offer(customer, common_state, vendor_specific_state, vendor_actions, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		customer.generate_purchase_probabilities_from_offer(customer, common_state, vendor_specific_state, vendor_actions)
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
