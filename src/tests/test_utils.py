import os
from importlib import reload
from unittest.mock import mock_open, patch

import pytest

from .context import utils


# Helper function that returns a mock config.json file/string with the given values
def create_mock_json(episode_size='20', learning_rate='1e-6', max_price='15', max_quality='100', number_of_customers='30', production_price='5'):
	return '{\n\t"episode_size": ' + episode_size + ',\n' + \
		'\t"learning_rate": ' + learning_rate + ',\n' + \
		'\t"max_price": ' + max_price + ',\n' + \
		'\t"max_quality": ' + max_quality + ',\n' + \
		'\t"number_of_customers": ' + number_of_customers + ',\n' + \
		'\t"production_price": ' + production_price + '\n}'


# Helper function that builds a mock config.json file/string that is missing a specified line
def create_mock_json_with_missing_line(number, json=create_mock_json()):
	lines = json.split('\n')
	final_lines = lines[0:number + 1]
	final_lines += lines[number + 2:len(lines)]
	final_lines[-2] = final_lines[-2].replace(',', '')
	return '\n'.join(final_lines)


# Helper function to test if the mock_file is setup correctly
def check_mock_file(mock_file, json=create_mock_json()):
	path = os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
	assert (open(path).read() == json)
	mock_file.assert_called_with(path)
	utils.config = utils.load_config(path)


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message
odd_number_of_customers = (create_mock_json('50', '1e-4', '50', '80', '21', '10'), 'number_of_customers should be even and positive')
negative_number_of_customers = (create_mock_json('50', '1e-4', '50', '80', '-10', '10'), 'number_of_customers should be even and positive')
learning_rate_larger_one = (create_mock_json('50', '1.5', '50', '80', '20', '10'), 'learning_rate should be between 0 and 1 (excluded)')
neg_learning_rate = (create_mock_json('50', '0', '50', '80', '20', '10'), 'learning_rate should be between 0 and 1 (excluded)')
prod_price_higher_max_price = (create_mock_json('50', '1e-5', '10', '80', '20', '50'), 'production_price needs to smaller than max_price and positive or zero')
neg_prod_price = (create_mock_json('50', '1e-5', '50', '80', '20', '-10'), 'production_price needs to smaller than max_price and positive or zero')
neg_max_quality = (create_mock_json('20', '1e-6', '15', '-80', '30', '5'), 'max_quality should be positive')

# These tests are missing a line in the config file, the import should throw a specific error message
missing_episode_size = (create_mock_json_with_missing_line(0), 'your config is missing episode_size')
missing_learning_rate = (create_mock_json_with_missing_line(1), 'your config is missing learning_rate')
missing_max_price = (create_mock_json_with_missing_line(2), 'your config is missing max_price')
missing_max_quality = (create_mock_json_with_missing_line(3), 'your config is missing max_quality')
missing_number_of_customers = (create_mock_json_with_missing_line(4), 'your config is missing number_of_customers')
missing_prod_price = (create_mock_json_with_missing_line(5), 'your config is missing production_price')

# All pairs concerning themselves with invalid config.json values should be added to this array to get tested in test_invalid_values
array_testing = [
	odd_number_of_customers, negative_number_of_customers, learning_rate_larger_one, neg_learning_rate, prod_price_higher_max_price, neg_prod_price, neg_max_quality,
	missing_episode_size, missing_learning_rate, missing_max_price, missing_max_quality, missing_number_of_customers, missing_prod_price
]


# This defines how the tests are named. Usually they would be "test_invalid_values[whole_json_here]". This ensures they are named after the actual thing they are testing
def get_invalid_test_ids():
	return [
		'odd_number_of_customers', 'negative_number_of_customers', 'learning_rate_larger_one', 'neg_learning_rate', 'prod_price_higher_max_price', 'neg_prod_price', 'neg_max_quality',
		'missing_episode_size', 'missing_learning_rate', 'missing_max_price', 'missing_max_quality', 'missing_number_of_customers', 'missing_prod_price'
	]


# Test that checks that an invalid/broken config.json gets detected correctly
@pytest.mark.parametrize('json_values, expected_error_msg', array_testing, ids=get_invalid_test_ids())
def test_invalid_values(json_values, expected_error_msg):
	json = json_values
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		check_mock_file(mock_file, json)
		with pytest.raises(AssertionError) as assertion_info:
			reload(utils)
		assert expected_error_msg in str(assertion_info.value)


# mock format taken from: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth
# Test that checks if the config.json is read correctly
def test_reading_file_values():
	json = create_mock_json()
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		check_mock_file(mock_file)

		# Include utils again to make sure the file is read again
		reload(utils)

		# Test all imported values. Extend this test as new values get added!
		assert utils.EPISODE_LENGTH == 20
		assert utils.LEARNING_RATE == 1e-6
		assert utils.MAX_PRICE == 15
		assert utils.MAX_QUALITY == 100
		assert utils.NUMBER_OF_CUSTOMERS == 30
		assert utils.PRODUCTION_PRICE == 5

	# Test a second time with other values to ensure, that the values are read correctly
	json2 = create_mock_json('50', '1e-4', '50', '80', '20', '10')
	with patch('builtins.open', mock_open(read_data=json2)) as mock_file:
		check_mock_file(mock_file, json2)
		reload(utils)

		assert utils.EPISODE_LENGTH == 50
		assert utils.LEARNING_RATE == 1e-4
		assert utils.MAX_PRICE == 50
		assert utils.MAX_QUALITY == 80
		assert utils.NUMBER_OF_CUSTOMERS == 20
		assert utils.PRODUCTION_PRICE == 10
