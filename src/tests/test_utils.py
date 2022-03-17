from importlib import reload
from unittest.mock import patch

import numpy as np
import pytest

import configuration.hyperparameter_config as hyperparameter_config
import configuration.utils as ut
from monitoring.svg_manipulation import SVGManipulator

testcases_cartesian_product = [
	([2, 3, 4], [5, 6], [(2, 5), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6)]),
	([7, 5], [9, 4], [(7, 9), (7, 4), (5, 9), (5, 4)]),
	(['Hund', 'Katze'], ['Maus'], [('Hund', 'Maus'), ('Katze', 'Maus')]),
	([('a', 'b'), ('c', 'd')], [1, 2], [(('a', 'b'), 1), (('a', 'b'), 2), (('c', 'd'), 1), (('c', 'd'), 2)])
]

testcases_shuffle_quality = [1, 10, 100, 1000, 10000]

testcases_softmax = [
	(
		[3., 4., 1., 10., 3., -1.],
		[9.07848731E-4, 0.0024677887, 1.2286396E-4, 0.99557702, 9.07848731E-4, 1.66278295E-5]
	),
	(
		[1., 1.],
		[0.5, 0.5]
	)
]

testcases_write_dict_svg = [(
	0,
	{
		'customer/buy_nothing': 2,
		'state/in_circulation': 455,
		'state/in_storage': {'vendor_0': 30, 'vendor_1': 79},
		'actions/price_refurbished': {'vendor_0': 4, 'vendor_1': 3},
		'actions/price_new': {'vendor_0': 6, 'vendor_1': 4},
		'owner/throw_away': 0,
		'owner/rebuys': {'vendor_0': 6, 'vendor_1': 7},
		'profits/rebuy_cost': {'vendor_0': -12, 'vendor_1': -5},
		'customer/purchases_refurbished': {'vendor_0': 1, 'vendor_1': 7},
		'customer/purchases_new': {'vendor_0': 5, 'vendor_1': 5},
		'profits/by_selling_refurbished': {'vendor_0': 4, 'vendor_1': 15},
		'profits/by_selling_new': {'vendor_0': 15, 'vendor_1': 5},
		'profits/storage_cost': {'vendor_0': -3.5, 'vendor_1': -7.9},
		'actions/price_rebuy': {'vendor_0': 2, 'vendor_1': 1},
		'profits/all': {'vendor_0': 3.5, 'vendor_1': 7.1},
	},
	{
		'customer/buy_nothing': 2,
		'state/in_circulation': 455,
		'state/in_storage': {'vendor_0': 30, 'vendor_1': 79},
		'actions/price_refurbished': {'vendor_0': 4, 'vendor_1': 3},
		'actions/price_new': {'vendor_0': 6, 'vendor_1': 4},
		'owner/throw_away': 0,
		'owner/rebuys': {'vendor_0': 6, 'vendor_1': 7},
		'profits/rebuy_cost': {'vendor_0': -12, 'vendor_1': -5},
		'customer/purchases_refurbished': {'vendor_0': 1, 'vendor_1': 7},
		'customer/purchases_new': {'vendor_0': 5, 'vendor_1': 5},
		'profits/by_selling_refurbished': {'vendor_0': 4, 'vendor_1': 15},
		'profits/by_selling_new': {'vendor_0': 15, 'vendor_1': 5},
		'profits/storage_cost': {'vendor_0': -3.5, 'vendor_1': -7.9},
		'actions/price_rebuy': {'vendor_0': 2, 'vendor_1': 1},
		'profits/all': {'vendor_0': 3.5, 'vendor_1': 7.1},
	},
	{
		'simulation_name': 'Market Simulation',
		'simulation_episode_length': '50',
		'simulation_current_episode': '1',
		'consumer_total_arrivals': '20',
		'consumer_total_sales': '18',
		'a_competitor_name': 'vendor_0',
		'a_throw_away': '0',
		'a_garbage': '0',
		'a_inventory': '30',
		'a_profit': '3.5',
		'a_price_new': '7',
		'a_price_used': '5',
		'a_rebuy_price': '3',
		'a_repurchases': '6',
		'a_resource_cost': '3',
		'a_resources_in_use': '455',
		'a_sales_new': '5',
		'a_sales_used': '1',
		'b_competitor_name': 'vendor_1',
		'b_inventory': '79',
		'b_profit': '7.1',
		'b_price_new': '5',
		'b_price_used': '4',
		'b_rebuy_price': '2',
		'b_repurchases': '7',
		'b_resource_cost': '3',
		'b_sales_new': '5',
		'b_sales_used': '7',
	}
)]


def test_ensure_results_folders_exist():
	pass


@pytest.mark.parametrize('max_quality', testcases_shuffle_quality)
def test_shuffle_quality(max_quality: int):
	with patch('configuration.hyperparameter_config.config') as config:
		config = hyperparameter_config.HyperparameterConfig()
		config.max_quality = max_quality
		quality = ut.shuffle_quality()
		assert quality <= max_quality and quality >= 1


@pytest.mark.parametrize('input_array, expected', testcases_softmax)
def test_softmax(input_array: np.array, expected: np.array):
	assert np.allclose(ut.softmax(input_array), expected)


def test_shuffle_from_probabilities():
	pass


@pytest.mark.parametrize('list_a, list_b, expected', testcases_cartesian_product)
def test_cartesian_product(list_a, list_b, expected):
	assert ut.cartesian_product(list_a, list_b) == expected


def test_write_dict_to_tensorboard():
	pass


def test_divide_content_of_dict():
	pass


def test_add_content_of_two_dicts():
	pass


@pytest.mark.parametrize('episode, episode_dictionary, cumulated_dictionary, expected',
	testcases_write_dict_svg)
def test_write_content_of_dict_to_overview_svg(
		episode: int,
		episode_dictionary: dict,
		cumulated_dictionary: dict,
		expected: dict):

	with patch('monitoring.svg_manipulation.SVGManipulator.write_dict_to_svg') as mock_write_dict_to_svg:
		ut.write_content_of_dict_to_overview_svg(SVGManipulator(), episode, episode_dictionary, cumulated_dictionary)
		mock_write_dict_to_svg.assert_called_once_with(target_dictionary=expected)


def import_config() -> hyperparameter_config.HyperparameterConfig:
	"""
	Reload the hyperparameter_config file to update the config variable with the mocked values.

	Returns:
		HyperparameterConfig: The config object.
	"""
	reload(hyperparameter_config)
	return hyperparameter_config.config
