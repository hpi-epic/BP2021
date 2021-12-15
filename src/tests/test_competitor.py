import pytest
from numpy import random

# from .context import competitor
# from .context import Competitor
from .context import CompetitorJust2Players as C2Players
from .context import CompetitorLinearRatio1 as CLinear1
from .context import CompetitorRandom as CRandom
from .context import utils as ut


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer():
	return [random.randint(1, ut.MAX_QUALITY), random.randint(1, ut.MAX_PRICE), random.randint(1, ut.MAX_QUALITY)]


def get_competitor_pricing_ids():
	return [
		'Linear1', 'Random', '2Players'
	]


array_competitor_pricing = [(CLinear1, random_offer()), (CRandom, random_offer()), (C2Players, random_offer())]


# Test the policy()-function of the different competitors
@pytest.mark.parametrize('competitor, state', array_competitor_pricing, ids=get_competitor_pricing_ids())
def test_policy(competitor, state):
	competitor.__init__(competitor)
	assert ut.PRODUCTION_PRICE == 2
	if competitor is CLinear1:
		assert ut.PRODUCTION_PRICE + 1 <= competitor.policy(competitor, state) < ut.MAX_PRICE
	if competitor is CRandom:
		assert ut.PRODUCTION_PRICE + 1 <= competitor.policy(competitor, state) < ut.MAX_PRICE
	if competitor is C2Players:
		assert ut.PRODUCTION_PRICE <= competitor.policy(competitor, state) < ut.MAX_PRICE
