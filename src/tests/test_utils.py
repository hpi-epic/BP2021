# import json
# from unittest.mock import MagicMock, patch

from .context import utils

# import pytest


# WIP: https://stackoverflow.com/questions/51138834/how-to-use-mock-open-with-json-load

# Perhaps take a look at: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth ?


def test_reading_file_values():
    # p1 = patch('builtins.open', MagicMock())

    # m = MagicMock(
    # 		side_effect=[
    # 				{
    # 						'episode_size': 40,
    # 						'learning_rate': 1e-5,
    # 						'max_price': 20,
    # 						'max_quality': 100,
    # 						'production_price': 10,
    # 						'number_of_customers': 20,
    # 				}
    # 		]
    # )
    # p2 = patch('json.load', m)

    # with p1 as p_open:
    # 	with p2 as p_json_load:
    # 		config_file = open('config.json')
    # 		print(json.load(config_file))
    assert utils.MAX_QUALITY == 100
