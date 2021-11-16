import pytest
from numpy import random

# from .context import sim_market, customer
from .context import ClassicScenario as SClassic
from .context import CustomerLinear as CLinear
from .context import MultiCompetitorScenario as SMulti


def random_state(market_scenario):
	ins = market_scenario()
	ins.reset()
	return ins.full_view(random.randint(1, 29))


# mark.parametrize can be used to run the same test with different parameters
# Test the LinearCustomer in the different Market Scenarios
@pytest.mark.parametrize('offers, expectedSize', [(random_state(SClassic), 4), (random_state(SMulti), 8)])
def test_linear_customer_action_range(offers, expectedSize):
	assert offers.size == expectedSize
	assert 0 <= CLinear.buy_object(CLinear, offers) <= expectedSize - 1
