import pytest
from numpy import random

from .context import ClassicScenario as SClassic
from .context import CustomerCircular as CCircular
from .context import CustomerLinear as CLinear
from .context import MultiCompetitorScenario as SMulti
from .context import customer


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer(market_scenario):
	ins = market_scenario()
	ins.reset()
	return ins.generate_offer(random.randint(1, 29))


# Test the Customer parent class, i.e. make sure it cannot be used
def test_customer_parent_class():
	with pytest.raises(AssertionError) as assertion_info:
		customer.Customer.buy_object(CLinear, random_offer(SClassic))
	assert str(assertion_info.value) == 'This class should not be used.'


array_customer_action_range = [
	(CLinear, random_offer(SClassic), 4), (CLinear, random_offer(SMulti), 8), (CCircular, random_offer(SClassic), 4), (CCircular, random_offer(SMulti), 8)
]


# mark.parametrize can be used to run the same test with different parameters
# Test the different Customers in the different Market Scenarios
@pytest.mark.parametrize('customer, offers, expectedSize', array_customer_action_range)
def test_customer_action_range(customer, offers, expectedSize):
	assert len(offers) == expectedSize
	buy_decisions = customer.buy_object(customer, offers)
	assert 0 <= buy_decisions[0] <= expectedSize - 1
	if customer is CLinear:
		assert buy_decisions[1] is None
	elif customer is CCircular:
		assert ((buy_decisions[1] is None) or (buy_decisions[1] == 1))
