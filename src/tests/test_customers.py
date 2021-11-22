import pytest
from numpy import random

# from .context import sim_market, customer
from .context import ClassicScenario as SClassic
from .context import CustomerLinear as CLinear
from .context import MultiCompetitorScenario as SMulti


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer(market_scenario):
	ins = market_scenario()
	ins.reset()
	return ins.generate_offer(random.randint(1, 29))

def get_linear_customer_action_range_ids():
	return [
		'ClassicScenario', 'MultiCompetitorScenario'
	]


# mark.parametrize can be used to run the same test with different parameters
# Test the LinearCustomer in the different Market Scenarios
@pytest.mark.parametrize('offers, expectedSize', [(random_offer(SClassic), 4), (random_offer(SMulti), 8)], ids = get_linear_customer_action_range_ids())
def test_linear_customer_action_range(offers, expectedSize):
	assert len(offers) == expectedSize
	buy_decisions = CLinear.buy_object(CLinear, offers)
	assert 0 <= buy_decisions[0] <= expectedSize - 1
	assert buy_decisions[1] is None
