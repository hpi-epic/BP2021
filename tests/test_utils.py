import configuration.utils as ut
import pytest

testcases = [
	([2, 3, 4], [5, 6], [(2, 5), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6)]),
	([7, 5], [9, 4], [(7, 9), (7, 4), (5, 9), (5, 4)]),
	(['Hund', 'Katze'], ['Maus'], [('Hund', 'Maus'), ('Katze', 'Maus')]),
	([('a', 'b'), ('c', 'd')], [1, 2], [(('a', 'b'), 1), (('a', 'b'), 2), (('c', 'd'), 1), (('c', 'd'), 2)])
]


@pytest.mark.parametrize('list_a, list_b, expected', testcases)
def test_cartesian_product(list_a, list_b, expected):
	assert ut.cartesian_product(list_a, list_b) == expected
