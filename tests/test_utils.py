from importlib import reload
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
import utils_tests as ut_t

import recommerce.configuration.hyperparameter_config as hyperparameter_config
import recommerce.configuration.utils as ut
from recommerce.monitoring.svg_manipulation import SVGManipulator


def teardown_module(module):
	reload(hyperparameter_config)
	reload(ut)


def import_config() -> hyperparameter_config.HyperparameterConfig:
	"""
	Reload the hyperparameter_config file to update the config variable with the mocked values.
	Returns:
		HyperparameterConfig: The config object.
	"""
	reload(hyperparameter_config)
	return hyperparameter_config.config


testcases_shuffle_quality = [1, 10, 100, 1000]


@pytest.mark.parametrize('max_quality', testcases_shuffle_quality)
def test_shuffle_quality(max_quality: int):
	mock_json = ut_t.create_hyperparameter_mock_json(sim_market=ut_t.create_hyperparameter_mock_json_sim_market(max_quality=str(max_quality)))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		import_config()
		reload(ut)
		quality = ut.shuffle_quality()
		assert quality <= max_quality and quality >= 1


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


@pytest.mark.parametrize('input_array, expected', testcases_softmax)
def test_softmax(input_array: np.array, expected: np.array):
	assert np.allclose(ut.softmax(input_array), expected)


testcases_shuffle_from_probabilities = [
	np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
	np.array([0.1, 0.2, 0.3, 0.3, 0.01, 0.05, 0.04]),
	np.array([0., 1.])
	]


@pytest.mark.parametrize('probabilities', testcases_shuffle_from_probabilities)
def test_shuffle_from_probabilities(probabilities: np.array):
	assert ut.shuffle_from_probabilities(probabilities) < len(probabilities)


testcases_cartesian_product = [
	([2, 3, 4], [5, 6], [(2, 5), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6)]),
	([7, 5], [9, 4], [(7, 9), (7, 4), (5, 9), (5, 4)]),
	(['Hund', 'Katze'], ['Maus'], [('Hund', 'Maus'), ('Katze', 'Maus')]),
	([('a', 'b'), ('c', 'd')], [1, 2], [(('a', 'b'), 1), (('a', 'b'), 2), (('c', 'd'), 1), (('c', 'd'), 2)])
]


@pytest.mark.parametrize('list_a, list_b, expected', testcases_cartesian_product)
def test_cartesian_product(list_a, list_b, expected):
	assert ut.cartesian_product(list_a, list_b) == expected


testcases_write_dict_tensorboard = [
	({'value_A': 1, 'value_B': 100}, 10, False),
	({'value_A': {1: 10, 2: 20}, 'value_B': {1: 9, 2: 19}}, 11, True)]


@pytest.mark.parametrize('dictionary, counter, is_cumulative', testcases_write_dict_tensorboard)
def test_write_dict_to_tensorboard(dictionary: dict, counter: int, is_cumulative: bool):

	mock_writer = Mock()

	ut.write_dict_to_tensorboard(mock_writer, dictionary, counter, is_cumulative)

	for name, content in dictionary.items():
		if is_cumulative:
			mock_writer.add_scalars.assert_any_call(f'cumulated_{name}', content, counter)
		else:
			mock_writer.add_scalar.assert_any_call(name, content, counter)


# contains two dicts with the same keys, the first one is the dict to divide by 2, the second one contains the expected result
testcases_divide_content_of_dict = [(	{
		'customer/buy_nothing': 2,
		'state/in_circulation': 454,
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
	2,
	{
		'customer/buy_nothing': 1.,
		'state/in_circulation': 227.,
		'state/in_storage': {'vendor_0': 15.0, 'vendor_1': 39.5},
		'actions/price_refurbished': {'vendor_0': 2., 'vendor_1': 1.5},
		'actions/price_new': {'vendor_0': 3., 'vendor_1': 2.},
		'owner/throw_away': 0.,
		'owner/rebuys': {'vendor_0': 3., 'vendor_1': 3.5},
		'profits/rebuy_cost': {'vendor_0': -6., 'vendor_1': -2.5},
		'customer/purchases_refurbished': {'vendor_0': 0.5, 'vendor_1': 3.5},
		'customer/purchases_new': {'vendor_0': 2.5, 'vendor_1': 2.5},
		'profits/by_selling_refurbished': {'vendor_0': 2, 'vendor_1': 7.5},
		'profits/by_selling_new': {'vendor_0': 7.5, 'vendor_1': 2.5},
		'profits/storage_cost': {'vendor_0': -1.75, 'vendor_1': -3.95},
		'actions/price_rebuy': {'vendor_0': 1., 'vendor_1': 0.5},
		'profits/all': {'vendor_0': 1.75, 'vendor_1': 3.55},
	})]


@pytest.mark.parametrize('input_dict, divisor, expected_dict', testcases_divide_content_of_dict)
def test_divide_content_of_dict(input_dict: dict, divisor: float, expected_dict: dict):
	assert ut.divide_content_of_dict(input_dict, divisor) == expected_dict


# contains three dicts with the same keys, the third is the sum of the first two key by key
testcases_add_content_dicts = [(
	{
		'customer/buy_nothing': 1.,
		'state/in_circulation': 227.,
		'state/in_storage': {'vendor_0': 15.0, 'vendor_1': 39.5},
		'actions/price_refurbished': {'vendor_0': 2., 'vendor_1': 1.5},
		'actions/price_new': {'vendor_0': 3., 'vendor_1': 2.},
		'owner/throw_away': 0.,
		'owner/rebuys': {'vendor_0': 3., 'vendor_1': 3.5},
		'profits/rebuy_cost': {'vendor_0': -6., 'vendor_1': -2.5},
		'customer/purchases_refurbished': {'vendor_0': 0.5, 'vendor_1': 3.5},
		'customer/purchases_new': {'vendor_0': 2.5, 'vendor_1': 2.5},
		'profits/by_selling_refurbished': {'vendor_0': 2, 'vendor_1': 7.5},
		'profits/by_selling_new': {'vendor_0': 7.5, 'vendor_1': 2.5},
		'profits/storage_cost': {'vendor_0': -1.75, 'vendor_1': -3.95},
		'actions/price_rebuy': {'vendor_0': 1., 'vendor_1': 0.5},
		'profits/all': {'vendor_0': 1.75, 'vendor_1': 3.55},
	}, {
		'customer/buy_nothing': 2.,
		'state/in_circulation': 228.,
		'state/in_storage': {'vendor_0': 16.0, 'vendor_1': 40.5},
		'actions/price_refurbished': {'vendor_0': 3., 'vendor_1': 2.5},
		'actions/price_new': {'vendor_0': 4., 'vendor_1': 3.},
		'owner/throw_away': 1.,
		'owner/rebuys': {'vendor_0': 4., 'vendor_1': 4.5},
		'profits/rebuy_cost': {'vendor_0': -5., 'vendor_1': -1.5},
		'customer/purchases_refurbished': {'vendor_0': 1.5, 'vendor_1': 4.5},
		'customer/purchases_new': {'vendor_0': 3.5, 'vendor_1': 3.5},
		'profits/by_selling_refurbished': {'vendor_0': 3, 'vendor_1': 8.5},
		'profits/by_selling_new': {'vendor_0': 8.5, 'vendor_1': 3.5},
		'profits/storage_cost': {'vendor_0': -0.75, 'vendor_1': -4.95},
		'actions/price_rebuy': {'vendor_0': 2., 'vendor_1': 1.5},
		'profits/all': {'vendor_0': 2.75, 'vendor_1': 4.55}
	}, {
		'customer/buy_nothing': 3,
		'state/in_circulation': 455,
		'state/in_storage': {'vendor_0': 31, 'vendor_1': 80},
		'actions/price_refurbished': {'vendor_0': 5, 'vendor_1': 4},
		'actions/price_new': {'vendor_0': 7, 'vendor_1': 5},
		'owner/throw_away': 1,
		'owner/rebuys': {'vendor_0': 7, 'vendor_1': 8},
		'profits/rebuy_cost': {'vendor_0': -11, 'vendor_1': -4},
		'customer/purchases_refurbished': {'vendor_0': 2, 'vendor_1': 8},
		'customer/purchases_new': {'vendor_0': 6, 'vendor_1': 6},
		'profits/by_selling_refurbished': {'vendor_0': 5, 'vendor_1': 16},
		'profits/by_selling_new': {'vendor_0': 16, 'vendor_1': 6},
		'profits/storage_cost': {'vendor_0': -2.5, 'vendor_1': -8.9},
		'actions/price_rebuy': {'vendor_0': 3, 'vendor_1': 2},
		'profits/all': {'vendor_0': 4.5, 'vendor_1': 8.1},
	})]


@pytest.mark.parametrize('dict_a, dict_b, expected_dict', testcases_add_content_dicts)
def test_add_content_of_two_dicts(dict_a: dict, dict_b: dict, expected_dict: dict):
	assert ut.add_content_of_two_dicts(dict_a, dict_b) == expected_dict


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


@pytest.mark.parametrize('episode, episode_dictionary, cumulated_dictionary, expected', testcases_write_dict_svg)
def test_write_content_of_dict_to_overview_svg(
		episode: int,
		episode_dictionary: dict,
		cumulated_dictionary: dict,
		expected: dict):
	mock_json = (ut_t.create_hyperparameter_mock_json(
		sim_market=ut_t.create_hyperparameter_mock_json_sim_market(episode_length='50', number_of_customers='20', production_price='3')))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		import_config()
		reload(ut)
		with patch('recommerce.monitoring.svg_manipulation.SVGManipulator.write_dict_to_svg') as mock_write_dict_to_svg:
			ut.write_content_of_dict_to_overview_svg(SVGManipulator(), episode, episode_dictionary, cumulated_dictionary)
		mock_write_dict_to_svg.assert_called_once_with(target_dictionary=expected)
