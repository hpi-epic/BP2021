import json
from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import recommerce.configuration.hyperparameter_config as hyperparameter_config


def teardown_module(module):
	print('***TEARDOWN***')
	reload(hyperparameter_config)


# mock format taken from:
# https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth
# Test that checks if the config.json is read correctly
def test_reading_file_values():
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(sim_market=ut_t.create_hyperparameter_mock_dict_sim_market()))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)

		config = hyperparameter_config.HyperparameterConfigLoader.load('hyperparameter_config')

		assert config.max_storage == 100
		assert config.episode_length == 25
		assert config.max_price == 10
		assert config.max_quality == 50
		assert config.number_of_customers == 10
		assert config.production_price == 3
		assert config.storage_cost_per_product == 0.1

	# Test a second time with other values to ensure, that the values are read correctly
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(
		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(50, 50, 50, 80, 20, 10, 0.7)))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)

		config = hyperparameter_config.HyperparameterConfigLoader.load('hyperparameter_config')

		assert config.max_storage == 50
		assert config.episode_length == 50
		assert config.max_price == 50
		assert config.max_quality == 80
		assert config.number_of_customers == 20
		assert config.production_price == 10
		assert config.storage_cost_per_product == 0.7


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message
odd_number_of_customers = (ut_t.create_hyperparameter_mock_dict_sim_market(number_of_customers=21),
	'number_of_customers should be even and positive')
negative_number_of_customers = (ut_t.create_hyperparameter_mock_dict_sim_market(10, 50, 50, 80, -10, 10, 0.15),
	'number_of_customers should be even and positive')
prod_price_higher_max_price = (ut_t.create_hyperparameter_mock_dict_sim_market(10, 50, 10, 80, 20, 50, 0.15),
	'production_price needs to be smaller than max_price and >=0')
negative_production_price = (ut_t.create_hyperparameter_mock_dict_sim_market(10, 50, 50, 80, 20, -10, 0.15),
	'production_price needs to be smaller than max_price and >=0')
negative_max_quality = (ut_t.create_hyperparameter_mock_dict_sim_market(10, 20, 15, -80, 30, 5, 0.15),
	'max_quality should be positive')
non_negative_storage_cost = (ut_t.create_hyperparameter_mock_dict_sim_market(10, 20, 15, 80, 30, 5, -3.5),
	'storage_cost_per_product should be non-negative')

# These tests are missing a line in the config file, the import should throw a specific error message
missing_max_storage = (ut_t.remove_key('max_storage', ut_t.create_hyperparameter_mock_dict_sim_market()),
	'your config is missing max_storage')
missing_episode_length = (ut_t.remove_key('episode_length', ut_t.create_hyperparameter_mock_dict_sim_market()),
	'your config is missing episode_length')
missing_max_price = (ut_t.remove_key('max_price', ut_t.create_hyperparameter_mock_dict_sim_market()),
	'your config is missing max_price')
missing_max_quality = (ut_t.remove_key('max_quality', ut_t.create_hyperparameter_mock_dict_sim_market()),
	'your config is missing max_quality')
missing_number_of_customers = (ut_t.remove_key('number_of_customers', ut_t.create_hyperparameter_mock_dict_sim_market()),
	'your config is missing number_of_customers')
missing_production_price = (ut_t.remove_key('production_price', ut_t.create_hyperparameter_mock_dict_sim_market()),
	'your config is missing production_price')
missing_storage_cost = (ut_t.remove_key('storage_cost_per_product', ut_t.create_hyperparameter_mock_dict_sim_market()),
	'your config is missing storage_cost_per_product')

# All pairs concerning themselves with invalid config.json values should be added to this array to get tested in test_invalid_values
invalid_values_testcases = [
	odd_number_of_customers,
	negative_number_of_customers,
	prod_price_higher_max_price,
	negative_production_price,
	negative_max_quality,
	non_negative_storage_cost,
	missing_max_storage,
	missing_episode_length,
	missing_max_price,
	missing_max_quality,
	missing_number_of_customers,
	missing_production_price,
	missing_storage_cost
]


# Test that checks that an invalid/broken config.json gets detected correctly
@pytest.mark.parametrize('sim_market_json, expected_message', invalid_values_testcases)
def test_invalid_values(sim_market_json, expected_message):
	mock_json = json.dumps(ut_t.create_hyperparameter_mock_dict(sim_market=sim_market_json))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		with pytest.raises(AssertionError) as assertion_message:
			hyperparameter_config.HyperparameterConfigLoader.load('hyperparameter_config')
		assert expected_message in str(assertion_message.value)
