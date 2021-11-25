#!/usr/bin/env python3

# helper
import json
import os

import numpy as np

MAX_PRICE = None
MAX_QUALITY = None
MEAN_REWARD_BOUND = None
NUMBER_OF_CUSTOMERS = None
PRODUCTION_PRICE = None
EPISODE_LENGTH = None


config = {}


def load_config(
    path_sim_market=os.path.dirname(__file__) + os.sep + '..' + os.sep + 'config_sim_market.json'
):
    with open(path_sim_market) as config_file:
        return json.load(config_file)


config = load_config()

# ordered alphabetically in the config_sim_market.json
assert 'episode_size' in config, 'your config is missing episode_size'
assert 'max_price' in config, 'your config is missing max_price'
assert 'max_quality' in config, 'your config is missing max_quality'
assert 'number_of_customers' in config, 'your config is missing number_of_customers'
assert 'production_price' in config, 'your config is missing production_price'

EPISODE_LENGTH = int(config['episode_size'])

MAX_PRICE = int(config['max_price'])
MAX_QUALITY = int(config['max_quality'])
NUMBER_OF_CUSTOMERS = int(config['number_of_customers'])
PRODUCTION_PRICE = int(config['production_price'])


assert NUMBER_OF_CUSTOMERS > 0 and NUMBER_OF_CUSTOMERS % 2 == 0, 'number_of_customers should be even and positive'
assert PRODUCTION_PRICE <= MAX_PRICE and PRODUCTION_PRICE >= 0, 'production_price needs to smaller than max_price and positive or zero'
assert MAX_QUALITY > 0, 'max_quality should be positive'
assert MAX_PRICE > 0, 'max_price should be positive'
assert EPISODE_LENGTH > 0, 'episode_size should be positive'

MEAN_REWARD_BOUND = EPISODE_LENGTH * MAX_PRICE * NUMBER_OF_CUSTOMERS


def shuffle_quality():
    return min(max(int(np.random.normal(MAX_QUALITY / 2, 2 * MAX_QUALITY / 5)), 1), MAX_QUALITY)
