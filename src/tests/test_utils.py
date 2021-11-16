import os
from importlib import reload
from unittest.mock import mock_open, patch

import pytest

from .context import utils


def create_mock_json(episode_size='20', learning_rate='1e-6', max_price='15', max_quality='100', number_of_customers='30', production_price='5'):
	return '{\n\t"episode_size": ' + episode_size + ',\n' + \
		'\t"learning_rate": ' + learning_rate + ',\n' + \
		'\t"max_price": ' + max_price + ',\n' + \
		'\t"max_quality": ' + max_quality + ',\n' + \
		'\t"number_of_customers": ' + number_of_customers + ',\n' + \
		'\t"production_price": ' + production_price + '\n}'


def check_mock_file(mock_file, json=create_mock_json()):
	path = os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
	assert (open(path).read() == json)
	mock_file.assert_called_with(path)
	utils.config = utils.load_config(path)


# Thank you!!11elf: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth

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


# These tests have invalid values in their input file, the import should throw a specific assertion
odd_number_of_customers = (create_mock_json('50', '1e-4', '50', '80', '21', '10'), 'number_of_customers should be even and positive')
negative_number_of_customers = (create_mock_json('50', '1e-4', '50', '80', '-10', '10'), 'number_of_customers should be even and positive')
learning_rate_larger_one = (create_mock_json('50', '1.5', '50', '80', '20', '10'), 'learning_rate should be between 0 and 1 (excluded)')
neg_learning_rate = (create_mock_json('50', '0', '50', '80', '20', '10'), 'learning_rate should be between 0 and 1 (excluded)')
prod_price_higher_max_price = (create_mock_json('50', '1e-5', '10', '80', '20', '50'), 'production_price needs to smaller than max_price and positive or zero')
neg_prod_price = (create_mock_json('50', '1e-5', '50', '80', '20', '-10'), 'production_price needs to smaller than max_price and positive or zero')
neg_max_quality = (create_mock_json('20', '1e-6', '15', '-80', '30', '5'), 'max_quality should be positive')

array_testing = [odd_number_of_customers, negative_number_of_customers, learning_rate_larger_one, neg_learning_rate, prod_price_higher_max_price, neg_prod_price, neg_max_quality]


@pytest.mark.parametrize('json_values, expected_error_msg', array_testing)
def test_invalid_values(json_values, expected_error_msg):
	json = json_values
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		check_mock_file(mock_file, json)
		with pytest.raises(AssertionError) as assertion_info:
			reload(utils)
		assert expected_error_msg in str(assertion_info.value)
