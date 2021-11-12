from ..sim_market import Customer

ident_state = [15, 20, 15, 20]


# Check that Customers only choose actions between 0 and 2
def test_customer_action_range():
    assert 0 <= Customer.buy_object(ident_state) <= 2
