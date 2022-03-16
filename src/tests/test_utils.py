from importlib import reload
from unittest.mock import mock_open, patch

import numpy as np
import pytest

import configuration.hyperparameter_config as hyperparameter_config
import configuration.utils as ut
import tests.utils_tests as ut_t

testcases_cartesian_product = [
	([2, 3, 4], [5, 6], [(2, 5), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6)]),
	([7, 5], [9, 4], [(7, 9), (7, 4), (5, 9), (5, 4)]),
	(['Hund', 'Katze'], ['Maus'], [('Hund', 'Maus'), ('Katze', 'Maus')]),
	([('a', 'b'), ('c', 'd')], [1, 2], [(('a', 'b'), 1), (('a', 'b'), 2), (('c', 'd'), 1), (('c', 'd'), 2)])
]

testcases_shuffle_quality = [1, 10, 100, 1000]

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


def test_ensure_results_folders_exist():
	pass


@pytest.mark.parametrize('max_quality', testcases_shuffle_quality)
def test_shuffle_quality(max_quality: str):
	# with patch('configuration.hyperparameter_config.config.max_quality', max_quality):
	mock_json = (ut_t.create_hyperparameter_mock_json(
		sim_market=ut_t.create_hyperparameter_mock_json_sim_market(max_quality=str(max_quality))))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		import_config()
		reload(ut)
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


def test_write_content_of_dict_to_overview_svg():
	pass


def import_config() -> hyperparameter_config.HyperparameterConfig:
	"""
	Reload the hyperparameter_config file to update the config variable with the mocked values.

	Returns:
		HyperparameterConfig: The config object.
	"""
	reload(hyperparameter_config)
	return hyperparameter_config.config
