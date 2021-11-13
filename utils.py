#!/usr/bin/env python3

# helper
import numpy as np
import json

MAX_QUALITY = None
MAX_PRICE = None
PRODUCTION_PRICE = None
STEPS_PER_ROUND = None
MEAN_REWARD_BOUND = None 
NUMBER_OF_CUSTOMERS = None

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50000
LEARNING_RATE = None
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 75000
EPSILON_START = 1.0
EPSILON_FINAL = 0.1

config = {}

with open('config.json') as config_file:
	config = json.load(config_file)

assert('episode_size' in config)
assert('max_quality' in config)
assert('max_price' in config)
assert('production_price' in config)
assert('learning_rate' in config)
assert('number_of_customers' in config)

STEPS_PER_ROUND = int(config['episode_size'])
MAX_QUALITY = int(config['max_quality'])
MAX_PRICE = int(config['max_price'])
LEARNING_RATE = float(config['learning_rate'])
NUMBER_OF_CUSTOMERS = int(config['number_of_customers'])
PRODUCTION_PRICE = int(config['production_price'])


assert(NUMBER_OF_CUSTOMERS > 0 and NUMBER_OF_CUSTOMERS % 2 == 0)
assert(LEARNING_RATE > 0 and LEARNING_RATE < 1)
assert(PRODUCTION_PRICE <= MAX_PRICE and PRODUCTION_PRICE >= 0)
assert(MAX_QUALITY > 0)
assert(MAX_PRICE > 0)
assert(STEPS_PER_ROUND > 0)

MEAN_REWARD_BOUND = STEPS_PER_ROUND * MAX_PRICE * 20


def shuffle_quality():
	return min(max(int(np.random.normal(MAX_QUALITY/ 2, MAX_QUALITY/5)), 1), MAX_QUALITY)
