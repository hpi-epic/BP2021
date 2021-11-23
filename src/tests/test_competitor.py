import pytest

# from .context import Competitor
from .context import competitor


# Test the Competitor parent class, i.e. make sure it cannot be used
def test_customer_parent_class():
	comp = competitor.Competitor()
	with pytest.raises(AssertionError) as assertion_info:
		comp.give_competitors_price(comp, 0)
	assert str(assertion_info.value) == 'You must use a subclass of Competitor!'
