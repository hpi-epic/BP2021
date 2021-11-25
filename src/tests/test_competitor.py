import pytest
from numpy import random

# from .context import Competitor
from .context import CompetitorJust2Players as C2Players
from .context import CompetitorLinearRatio1 as CLinear1
from .context import CompetitorRandom as CRandom
from .context import competitor
from .context import utils as ut


# Helper function that creates a random offer (state that includes the agent's price) to test customer behaviour. This is dependent on the sim_market working!
def random_offer():
	return [random.randint(1, ut.MAX_PRICE), random.randint(1, ut.MAX_QUALITY), random.randint(1, ut.MAX_PRICE), random.randint(1, ut.MAX_QUALITY)]


# Test the Competitor parent class, i.e. make sure it cannot be used
def test_competitor_parent_class():
	comp = competitor.Competitor()
	with pytest.raises(AssertionError) as assertion_info:
		comp.give_competitors_price(comp, 0)
	assert str(assertion_info.value) == 'You must use a subclass of Competitor!'


def get_competitor_pricing_ids():
	return [
		'Linear1', 'Random', '2Players'
	]


array_competitor_pricing = [(CLinear1, random_offer(), 1), (CRandom, random_offer(), 1), (C2Players, random_offer(), 1)]


# Test the give_competitors_price()-function of the different competitors
@pytest.mark.parametrize('competitor, state, self_idx', array_competitor_pricing, ids=get_competitor_pricing_ids())
def test_give_competitors_price(competitor, state, self_idx):
	competitor.__init__(competitor)
	if competitor is CLinear1:
		# the 23 is a magic number in the give_competitors_price function, we should make this a constant defined somewhere
		assert ut.PRODUCTION_PRICE + 1 <= competitor.give_competitors_price(competitor, state, self_idx) <= 23
	if competitor is CRandom:
		assert ut.PRODUCTION_PRICE + 1 <= competitor.give_competitors_price(competitor, state, self_idx) <= ut.MAX_PRICE
	if competitor is C2Players:
		assert ut.PRODUCTION_PRICE <= competitor.give_competitors_price(competitor, state, self_idx) <= ut.MAX_PRICE
