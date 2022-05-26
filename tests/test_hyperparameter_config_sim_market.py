import json
import os
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager

market_config_file = os.path.join(PathManager.user_path, 'configuration_files', 'market_config.json')


# Test that checks if the config.json is read correctly
def test_reading_file_values():
	config = HyperparameterConfigLoader.load('market_config')

	assert config.max_storage == 100
	assert config.episode_length == 50
	assert config.max_price == 10
	assert config.max_quality == 50
	assert config.number_of_customers == 20
	assert config.production_price == 3
	assert config.storage_cost_per_product == 0.1


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message
odd_number_of_customers = (ut_t.replace_field_in_dict(ut_t.load_json(market_config_file), 'number_of_customers', 21),
	'number_of_customers should be even and positive')
negative_number_of_customers = (ut_t.replace_field_in_dict(ut_t.load_json(market_config_file), 'number_of_customers', -10),
	'number_of_customers should be even and positive')
negative_production_price = (ut_t.replace_field_in_dict(ut_t.load_json(market_config_file), 'production_price', -10),
	'production_price should be non-negative')
negative_max_quality = (ut_t.replace_field_in_dict(ut_t.load_json(market_config_file), 'max_quality', -80),
	'max_quality should be positive')
non_negative_storage_cost = (ut_t.replace_field_in_dict(ut_t.load_json(market_config_file), 'storage_cost_per_product', -3.5),
	'storage_cost_per_product should be non-negative')

# These tests are missing a line in the config file, the import should throw a specific error message
missing_max_storage = (ut_t.remove_key('max_storage', ut_t.load_json(market_config_file)),
	'your config is missing max_storage')
missing_episode_length = (ut_t.remove_key('episode_length', ut_t.load_json(market_config_file)),
	'your config is missing episode_length')
missing_max_price = (ut_t.remove_key('max_price', ut_t.load_json(market_config_file)),
	'your config is missing max_price')
missing_max_quality = (ut_t.remove_key('max_quality', ut_t.load_json(market_config_file)),
	'your config is missing max_quality')
missing_number_of_customers = (ut_t.remove_key('number_of_customers', ut_t.load_json(market_config_file)),
	'your config is missing number_of_customers')
missing_production_price = (ut_t.remove_key('production_price', ut_t.load_json(market_config_file)),
	'your config is missing production_price')
missing_storage_cost = (ut_t.remove_key('storage_cost_per_product', ut_t.load_json(market_config_file)),
	'your config is missing storage_cost_per_product')

# All pairs concerning themselves with invalid config.json values should be added to this array to get tested in test_invalid_values
invalid_values_testcases = [
	odd_number_of_customers,
	negative_number_of_customers,
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
@pytest.mark.parametrize('market_json, expected_message', invalid_values_testcases)
def test_invalid_values(market_json, expected_message):
	mock_json = json.dumps(market_json)
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		with pytest.raises(AssertionError) as assertion_message:
			HyperparameterConfigLoader.load('hyperparameter_config')
		assert expected_message in str(assertion_message.value)
