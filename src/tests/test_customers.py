import pytest

from .context import ClassicScenario as SClassic
from .context import CustomerCircular as CCircular
from .context import CustomerLinear as CLinear
from .context import MultiCompetitorScenario as SMulti
from .context import customer


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer(market_scenario):
	market = market_scenario()
	market.reset()
	return market.generate_offer(market.action_space.sample())


# Test the Customer parent class, i.e. make sure it cannot be used
def test_customer_parent_class():
	with pytest.raises(AssertionError) as assertion_info:
		customer.Customer.buy_object(CLinear, random_offer(SClassic))
	assert str(assertion_info.value) == 'This class should not be used.'


def get_customer_action_range_ids():
	return [
	]


array_customer_action_range = [
	(CLinear, random_offer(SClassic), 4), (CLinear, random_offer(SMulti), 8), (CCircular, random_offer(SClassic), 4), (CCircular, random_offer(SMulti), 8)
]


# mark.parametrize can be used to run the same test with different parameters
# Test the different Customers in the different Market Scenarios
@pytest.mark.parametrize('customer, offers, expectedSize', array_customer_action_range, ids=get_customer_action_range_ids())
def test_customer_action_range(customer, offers, expectedSize):
	assert len(offers) == expectedSize
	customer.set_probabilities_from_offers(customer, offers)
	buy_decisions = customer.buy_object(customer, offers)
	assert 0 <= buy_decisions <= expectedSize - 1
