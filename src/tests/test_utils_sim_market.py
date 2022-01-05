from importlib import reload
from unittest.mock import mock_open, patch

import pytest

import tests.utils_tests as ut_t
import utils_sim_market as ut


# mock format taken from: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth
# Test that checks if the config.json is read correctly
def test_reading_file_values():
	json = ut_t.create_mock_json_sim_market()
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)

		# Include utils again to make sure the file is read again
		reload(ut)

		# Test all imported values. Extend this test as new values get added!
		assert len(ut.config) == 5, 'utils has more or less values than expected. Check this test for the missing values'
		assert ut.EPISODE_LENGTH == 20
		assert ut.MAX_PRICE == 15
		assert ut.MAX_QUALITY == 100
		assert ut.NUMBER_OF_CUSTOMERS == 30
		assert ut.PRODUCTION_PRICE == 5

	# Test a second time with other values to ensure, that the values are read correctly
	json2 = ut_t.create_mock_json_sim_market('50', '50', '80', '20', '10')
	with patch('builtins.open', mock_open(read_data=json2)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json2)
		reload(ut)

		assert ut.EPISODE_LENGTH == 50
		assert ut.MAX_PRICE == 50
		assert ut.MAX_QUALITY == 80
		assert ut.NUMBER_OF_CUSTOMERS == 20
		assert ut.PRODUCTION_PRICE == 10


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message
odd_number_of_customers = (ut_t.create_mock_json_sim_market('50', '50', '80', '21', '10'), 'number_of_customers should be even and positive')
negative_number_of_customers = (ut_t.create_mock_json_sim_market('50', '50', '80', '-10', '10'), 'number_of_customers should be even and positive')
prod_price_higher_max_price = (ut_t.create_mock_json_sim_market('50', '10', '80', '20', '50'), 'production_price needs to smaller than max_price and positive or zero')
neg_prod_price = (ut_t.create_mock_json_sim_market('50', '50', '80', '20', '-10'), 'production_price needs to smaller than max_price and positive or zero')
neg_max_quality = (ut_t.create_mock_json_sim_market('20', '15', '-80', '30', '5'), 'max_quality should be positive')

# These tests are missing a line in the config file, the import should throw a specific error message
missing_episode_size = (ut_t.remove_line(0, ut_t.create_mock_json_sim_market()), 'your config is missing episode_size')
missing_max_price = (ut_t.remove_line(1, ut_t.create_mock_json_sim_market()), 'your config is missing max_price')
missing_max_quality = (ut_t.remove_line(2, ut_t.create_mock_json_sim_market()), 'your config is missing max_quality')
missing_number_of_customers = (ut_t.remove_line(3, ut_t.create_mock_json_sim_market()), 'your config is missing number_of_customers')
missing_prod_price = (ut_t.remove_line(4, ut_t.create_mock_json_sim_market()), 'your config is missing production_price')

# All pairs concerning themselves with invalid config.json values should be added to this array to get tested in test_invalid_values
array_invalid_values = [
	odd_number_of_customers, negative_number_of_customers, prod_price_higher_max_price, neg_prod_price, neg_max_quality,
	missing_episode_size, missing_max_price, missing_max_quality, missing_number_of_customers, missing_prod_price
]


# This defines how the tests are named. Usually they would be "test_invalid_values[whole_json_here]". This ensures they are named after the actual thing they are testing
def get_invalid_test_ids():
	return [
		'odd_number_of_customers', 'negative_number_of_customers', 'prod_price_higher_max_price', 'neg_prod_price', 'neg_max_quality',
		'missing_episode_size', 'missing_max_price', 'missing_max_quality', 'missing_number_of_customers', 'missing_prod_price'
	]


# Test that checks that an invalid/broken config.json gets detected correctly
@pytest.mark.parametrize('json_values, expected_error_msg', array_invalid_values, ids=get_invalid_test_ids())
def test_invalid_values(json_values, expected_error_msg):
	json = json_values
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file_sim_market(mock_file, json)
		with pytest.raises(AssertionError) as assertion_info:
			reload(ut)
		assert expected_error_msg in str(assertion_info.value)
