from importlib import reload
from unittest.mock import mock_open, patch

import pytest

import configuration.config as config
import tests.utils_tests as ut_t


def teardown_module(module):
	print('***TEARDOWN***')
	reload(config)


# mock format taken from: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth
# Test that checks if the config.json is read correctly
def test_reading_file_values():
	json = ut_t.create_mock_json(sim_market=ut_t.create_mock_json_sim_market())
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)

		# Include utils again to make sure the file is read again
		reload(config)

		# Test all imported values. Extend this test as new values get added!
		assert len(config.config) == 2, 'the config is being tested for "rl" and "sim_market". Has another type been added?'
		assert len(config.config['sim_market']) == 5, 'config["sim_market"] has more or less values than expected. Check this test for the missing values'
		assert config.EPISODE_LENGTH == 20
		assert config.MAX_PRICE == 15
		assert config.MAX_QUALITY == 100
		assert config.NUMBER_OF_CUSTOMERS == 30
		assert config.PRODUCTION_PRICE == 5

	# Test a second time with other values to ensure, that the values are read correctly
	json2 = ut_t.create_mock_json(sim_market=ut_t.create_mock_json_sim_market('50', '50', '80', '20', '10'))
	with patch('builtins.open', mock_open(read_data=json2)) as mock_file:
		ut_t.check_mock_file(mock_file, json2)
		reload(config)

		assert config.EPISODE_LENGTH == 50
		assert config.MAX_PRICE == 50
		assert config.MAX_QUALITY == 80
		assert config.NUMBER_OF_CUSTOMERS == 20
		assert config.PRODUCTION_PRICE == 10


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message
odd_number_of_customers = (ut_t.create_mock_json_sim_market('50', '50', '80', '21', '10'), 'number_of_customers should be even and positive')
negative_number_of_customers = (ut_t.create_mock_json_sim_market('50', '50', '80', '-10', '10'), 'number_of_customers should be even and positive')
prod_price_higher_max_price = (ut_t.create_mock_json_sim_market('50', '10', '80', '20', '50'), 'production_price needs to smaller than max_price and positive or zero')
negative_production_price = (ut_t.create_mock_json_sim_market('50', '50', '80', '20', '-10'), 'production_price needs to smaller than max_price and positive or zero')
negative_max_quality = (ut_t.create_mock_json_sim_market('20', '15', '-80', '30', '5'), 'max_quality should be positive')

# These tests are missing a line in the config file, the import should throw a specific error message
missing_episode_size = (ut_t.remove_line(0, ut_t.create_mock_json_sim_market()), 'your config is missing episode_size')
missing_max_price = (ut_t.remove_line(1, ut_t.create_mock_json_sim_market()), 'your config is missing max_price')
missing_max_quality = (ut_t.remove_line(2, ut_t.create_mock_json_sim_market()), 'your config is missing max_quality')
missing_number_of_customers = (ut_t.remove_line(3, ut_t.create_mock_json_sim_market()), 'your config is missing number_of_customers')
missing_production_price = (ut_t.remove_line(4, ut_t.create_mock_json_sim_market()), 'your config is missing production_price')

# All pairs concerning themselves with invalid config.json values should be added to this array to get tested in test_invalid_values
invalid_values_testcases = [
	odd_number_of_customers, negative_number_of_customers, prod_price_higher_max_price, negative_production_price, negative_max_quality,
	missing_episode_size, missing_max_price, missing_max_quality, missing_number_of_customers, missing_production_price
]


# Test that checks that an invalid/broken config.json gets detected correctly
@pytest.mark.parametrize('sim_market_json, expected_message', invalid_values_testcases)
def test_invalid_values(sim_market_json, expected_message):
	json = ut_t.create_mock_json(sim_market=sim_market_json)
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		with pytest.raises(AssertionError) as assertion_message:
			reload(config)
		assert expected_message in str(assertion_message.value)
