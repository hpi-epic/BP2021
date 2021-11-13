import pytest

from .context import Customer, SimMarket


@pytest.fixture
def random_state():
    ins = SimMarket()
    return ins.reset()


# Check that Customers only choose actions between 0 and 2 given a random state
def test_customer_action_range(random_state):
    assert 0 <= Customer.buy_object(random_state) <= 2
