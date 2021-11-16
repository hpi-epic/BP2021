import os
from importlib import reload
from unittest.mock import mock_open, patch

from .context import utils


def create_mock_json(episode_size='20', learning_rate='1e-6', max_price='15', max_quality='100', number_of_customers='30', production_price='5'):
	return '{\n\t"episode_size": ' + episode_size + ',\n' + \
		'\t"learning_rate": ' + learning_rate + ',\n' + \
		'\t"max_price": ' + max_price + ',\n' + \
		'\t"max_quality": ' + max_quality + ',\n' + \
		'\t"number_of_customers": ' + number_of_customers + ',\n' + \
		'\t"production_price": ' + production_price + '\n}'


# Thank you!!11elf: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth

def test_reading_file_values():
	with patch(
		'builtins.open',
		mock_open(
			read_data=create_mock_json()),
	) as mock_file:
		assert(
			open(
				os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
			).read()
			== create_mock_json()
		)
		# check that the mock_file is read correctly when opening any file
		mock_file.assert_called_with(
			os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
		)
		utils.config = utils.load_config(
			os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
		)
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
	with patch(
		'builtins.open',
		mock_open(
			read_data=create_mock_json('50', '1e-4', '50', '80', '20', '10')
		),
	) as mock_file:
		assert (
			open(
				os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
			).read() == create_mock_json('50', '1e-4', '50', '80', '20', '10')
		)
		mock_file.assert_called_with(
			os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
		)
		utils.config = utils.load_config(
			os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
		)
		reload(utils)

		assert utils.EPISODE_LENGTH == 50
		assert utils.LEARNING_RATE == 1e-4
		assert utils.MAX_PRICE == 50
		assert utils.MAX_QUALITY == 80
		assert utils.NUMBER_OF_CUSTOMERS == 20
		assert utils.PRODUCTION_PRICE == 10


def test_invalid_values():
	pass
