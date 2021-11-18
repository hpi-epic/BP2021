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

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 200000
LEARNING_RATE = None
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 75000
EPSILON_START = 1.0
EPSILON_FINAL = 0.1

config = {}


def load_config(
    path=os.path.dirname(__file__) + os.sep + '..' + os.sep + 'config.json'
):
    with open(path) as config_file:
        return json.load(config_file)


config = load_config()

# ordered alphabetically in the config.json
assert 'episode_size' in config
assert 'learning_rate' in config
assert 'max_price' in config
assert 'max_quality' in config
assert 'number_of_customers' in config
assert 'production_price' in config

EPISODE_LENGTH = int(config['episode_size'])
LEARNING_RATE = float(config['learning_rate'])
MAX_PRICE = int(config['max_price'])
MAX_QUALITY = int(config['max_quality'])
NUMBER_OF_CUSTOMERS = int(config['number_of_customers'])
PRODUCTION_PRICE = int(config['production_price'])


assert NUMBER_OF_CUSTOMERS > 0 and NUMBER_OF_CUSTOMERS % 2 == 0
assert LEARNING_RATE > 0 and LEARNING_RATE < 1
assert PRODUCTION_PRICE <= MAX_PRICE and PRODUCTION_PRICE >= 0
assert MAX_QUALITY > 0
assert MAX_PRICE > 0
assert EPISODE_LENGTH > 0

MEAN_REWARD_BOUND = EPISODE_LENGTH * MAX_PRICE * 20


def shuffle_quality():
    return min(max(int(np.random.normal(MAX_QUALITY / 2, 2 * MAX_QUALITY / 5)), 1), MAX_QUALITY)
