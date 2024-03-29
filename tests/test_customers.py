import numpy as np
import pytest
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.customer as customer
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_customers import CustomerCircular
from recommerce.market.linear.linear_customers import CustomerLinear
from recommerce.market.sim_market import SimMarket

config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceMonopoly)


# Test the Customer parent class, i.e. make sure it cannot be used
def test_customer_parent_class():
	with pytest.raises(NotImplementedError) as assertion_message:
		customer.Customer.generate_purchase_probabilities_from_offer(CustomerLinear, *random_offer(linear_market.LinearEconomyDuopoly))
	assert 'This method is abstract. Use a subclass' in str(assertion_message.value)


# the following list contains invalid parameters for generate_purchase_probabilities_from_offer and the expected error messages
generate_purchase_probabilities_from_offer_testcases = [
	(CustomerLinear, [], [[12], [15]], [3, 5], 'common_state must be a np.ndarray'),
	(CustomerLinear, np.array([]), ([12], [15]), [3, 5], 'vendor_specific_state must be a list'),
	(CustomerLinear, np.array([]), [[12], [15]], (3, 5), 'vendor_actions must be a list'),
	(CustomerLinear, np.array([]), [[12]], [3, 5], 'they must have the same length'),
	(CustomerLinear, np.array([]), [[12], [15]], [3], 'they must have the same length'),
	(CustomerLinear, np.array([]), [], [], 'there must be at least one vendor'),
	(CustomerCircular, [], [[17], [23]], [[3, 6], [4, 7]], 'common_state must be a np.ndarray'),
	(CustomerLinear, np.array([]), ([17], [23]), [[3, 6], [4, 7]], 'vendor_specific_state must be a list'),
	(CustomerCircular, np.array([]), [[17], [23]], ([3, 6], [4, 7]), 'vendor_actions must be a list'),
	(CustomerCircular, np.array([]), [[17]], [[3, 6], [4, 7]], 'they must have the same length'),
	(CustomerCircular, np.array([]), [[17], [23]], [[3, 6]], 'they must have the same length'),
	(CustomerCircular, np.array([]), [], [], 'there must be at least one vendor'),
]


@pytest.mark.parametrize(
	'customer, common_state, vendor_specific_state, vendor_actions, expected_message',
	generate_purchase_probabilities_from_offer_testcases)
def test_generate_purchase_probabilities_from_offer(customer, common_state, vendor_specific_state, vendor_actions, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		customer.generate_purchase_probabilities_from_offer(customer, common_state, vendor_specific_state, vendor_actions)
	assert expected_message in str(assertion_message.value)


customer_action_range_testcases = [
	(CustomerLinear, linear_market.LinearEconomyDuopoly),
	(CustomerLinear, linear_market.LinearEconomyOligopoly),
	(CustomerCircular, circular_market.CircularEconomyMonopoly),
	(CustomerCircular, circular_market.CircularEconomyRebuyPriceMonopoly)
]


# Test the different Customers in the different Market Scenarios
@pytest.mark.parametrize('customer, market', customer_action_range_testcases)
def test_customer_action_range(customer, market):
	offers = random_offer(market)
	probability_distribution = customer.generate_purchase_probabilities_from_offer(customer, *offers)
	assert len(probability_distribution) == market(config=config_market)._get_number_of_vendors() * \
		(1 if issubclass(market, linear_market.LinearEconomy) else 2) + 1


def test_linear_higher_price_lower_purchase_probability():
	common_state, vendor_specific_state, vendor_actions = np.array([]), [[12], [12]], [3, 5]
	probability_distribution = CustomerLinear.generate_purchase_probabilities_from_offer(
		CustomerLinear, common_state, vendor_specific_state, vendor_actions)
	assert probability_distribution[1] > probability_distribution[2]


def test_linear_higher_quality_higher_purchase_probability():
	common_state, vendor_specific_state, vendor_actions = np.array([]), [[13], [12]], [3, 3]
	probability_distribution = CustomerLinear.generate_purchase_probabilities_from_offer(
		CustomerLinear, common_state, vendor_specific_state, vendor_actions)
	assert probability_distribution[1] > probability_distribution[2]


def test_equal_ratio_equal_purchase_probability():
	# In the following line: [3, 1] means prices [4, 2]
	common_state, vendor_specific_state, vendor_actions = np.array([]), [[16], [8]], [3, 1]
	probability_distribution = CustomerLinear.generate_purchase_probabilities_from_offer(
		CustomerLinear, common_state, vendor_specific_state, vendor_actions)
	assert probability_distribution[1] == probability_distribution[2]


def test_linear_lower_overall_price_lower_nothing_probability():
	common_state1, vendor_specific_state1, vendor_actions1 = np.array([]), [[15], [15]], [3, 3]
	probability_distribution1 = CustomerLinear.generate_purchase_probabilities_from_offer(
		CustomerLinear, common_state1, vendor_specific_state1, vendor_actions1)
	common_state2, vendor_specific_state2, vendor_actions2 = np.array([]), [[15], [15]], [4, 4]
	probability_distribution2 = CustomerLinear.generate_purchase_probabilities_from_offer(
		CustomerLinear, common_state2, vendor_specific_state2, vendor_actions2)
	print(probability_distribution1)
	print(probability_distribution2)
	assert probability_distribution1[0] < probability_distribution2[0]
	assert probability_distribution1[1] > probability_distribution2[1]
	assert probability_distribution1[2] > probability_distribution2[2]


def test_circular_higher_price_lower_purchase_probability():
	common_state, vendor_specific_state, vendor_actions = np.array([]), [[17], [23]], [[3, 6], [4, 5]]
	probability_distribution = CustomerCircular.generate_purchase_probabilities_from_offer(
		CustomerCircular, common_state, vendor_specific_state, vendor_actions)
	assert probability_distribution[1] > probability_distribution[3]
	assert probability_distribution[2] < probability_distribution[4]


def random_offer(marketplace: SimMarket):
	"""
	Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour.

	This is dependent on the sim_market working!

	Args:
		marketplace (SimMarket): The marketplace for which offers should be generated.
	"""
	marketplace = marketplace(config=config_market)
	marketplace.reset()
	marketplace.vendor_actions[0] = marketplace.action_space.sample()
	return marketplace._get_common_state_array(), marketplace.vendor_specific_state, marketplace.vendor_actions
