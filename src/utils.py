#!/usr/bin/env python3

# helper
import json
import os
import random

import numpy as np

MAX_STORAGE = 100
MAX_PRICE = None
MAX_QUALITY = None
MEAN_REWARD_BOUND = None
NUMBER_OF_CUSTOMERS = None
PRODUCTION_PRICE = None
EPISODE_LENGTH = None

STORAGE_COST_PER_PRODUCT = 0.5

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
assert 'episode_size' in config, 'your config is missing episode_size'
assert 'learning_rate' in config, 'your config is missing learning_rate'
assert 'max_price' in config, 'your config is missing max_price'
assert 'max_quality' in config, 'your config is missing max_quality'
assert 'number_of_customers' in config, 'your config is missing number_of_customers'
assert 'production_price' in config, 'your config is missing production_price'

EPISODE_LENGTH = int(config['episode_size'])
LEARNING_RATE = float(config['learning_rate'])
MAX_PRICE = int(config['max_price'])
MAX_QUALITY = int(config['max_quality'])
NUMBER_OF_CUSTOMERS = int(config['number_of_customers'])
PRODUCTION_PRICE = int(config['production_price'])


assert NUMBER_OF_CUSTOMERS > 0 and NUMBER_OF_CUSTOMERS % 2 == 0, 'number_of_customers should be even and positive'
assert LEARNING_RATE > 0 and LEARNING_RATE < 1, 'learning_rate should be between 0 and 1 (excluded)'
assert PRODUCTION_PRICE <= MAX_PRICE and PRODUCTION_PRICE >= 0, 'production_price needs to smaller than max_price and positive or zero'
assert MAX_QUALITY > 0, 'max_quality should be positive'
assert MAX_PRICE > 0, 'max_price should be positive'
assert EPISODE_LENGTH > 0, 'episode_size should be positive'

MEAN_REWARD_BOUND = EPISODE_LENGTH * MAX_PRICE * 20


def shuffle_quality():
    return min(max(int(np.random.normal(MAX_QUALITY / 2, 2 * MAX_QUALITY / 5)), 1), MAX_QUALITY)


# The following methods should be library calls in the future.
def softmax(preferences) -> np.array:
    exp_preferences = np.exp(preferences)
    return exp_preferences / sum(exp_preferences)


def shuffle_from_probabilities(probabilities) -> int:
    randomnumber = random.random()
    sum = 0
    for i, p in enumerate(probabilities):
        sum += p
        if randomnumber <= sum:
            return i
    return len(probabilities) - 1
