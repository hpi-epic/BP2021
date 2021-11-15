# import json
# from unittest.mock import MagicMock, patch

# import pytest

from .context import utils

# WIP: https://stackoverflow.com/questions/51138834/how-to-use-mock-open-with-json-load


def test_reading_file_values():
    # p1 = patch('builtins.open', MagicMock())

    # m = MagicMock(
    #     side_effect=[
    #         {
    #             'episode_size': 40,
    #             'learning_rate': 1e-5,
    #             'max_price': 20,
    #             'max_quality': 100,
    #             'production_price': 10,
    #             'number_of_customers': 20,
    #         }
    #     ]
    # )
    # p2 = patch('json.load', m)

    # with p1 as p_open:
    #     with p2 as p_json_load:
    #         f = open('config.json')
    #         print(json.load(f))
    assert utils.MAX_QUALITY == 100
