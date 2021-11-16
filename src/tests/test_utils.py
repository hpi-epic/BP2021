import os
from importlib import reload
from unittest.mock import mock_open, patch

from .context import utils

# Than you!!11elf: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth ?


def test_reading_file_values():
    with patch(
        'builtins.open',
        mock_open(
            read_data='''{"episode_size": 20,
                "learning_rate": 1e-6,
                "max_price": 15,
                "max_quality": 100,
                "production_price": 25,
                "number_of_customers": 30}'''
        ),
    ) as mock_file:
        assert (
            open(
                os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
            ).read()
            == '''{"episode_size": 20,
                "learning_rate": 1e-6,
                "max_price": 15,
                "max_quality": 100,
                "production_price": 25,
                "number_of_customers": 30}'''
        )
        mock_file.assert_called_with(
            os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
        )
        utils.config = utils.load_config(
            os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config.json'
        )
        reload(utils)

        assert utils.MAX_QUALITY == 100
