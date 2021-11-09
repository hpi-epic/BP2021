import numpy as np
import json
MAX_QUALITY = None
MAX_PRICE = None
PRODUCTION_PRICE = None
STEPS_PER_ROUND = None
MEAN_REWARD_BOUND = None 

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50000
LEARNING_RATE = None
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 75000
EPSILON_START = 1.0
EPSILON_FINAL = 0.1

def setup():
        global STEPS_PER_ROUND, MAX_QUALITY, MAX_PRICE, PRODUCTION_PRICE, MEAN_REWARD_BOUND, LEARNING_RATE
        config = {}
        
        with open('config.json') as config_file:
                config = json.load(config_file)
        
        assert('episode_size' in config)
        assert('max_quality' in config)
        assert('max_price' in config)
        assert('production_price' in config)
        assert('learning_rate' in config)

        STEPS_PER_ROUND = int(config['episode_size'])
        MAX_QUALITY = int(config['max_quality'])
        MAX_PRICE = int(config['max_price'])
        LEARNING_RATE = float(config['learning_rate'])

        if config['production_price'] == 0:
                PRODUCTION_PRICE = int(MAX_PRICE / 3)
        else:
                PRODUCTION_PRICE = int(config['production_price'])
        MEAN_REWARD_BOUND = STEPS_PER_ROUND * MAX_PRICE * 20
        assert(LEARNING_RATE > 0 and LEARNING_RATE < 1)
        assert(PRODUCTION_PRICE <= MAX_PRICE and PRODUCTION_PRICE > 0)
        assert(MAX_QUALITY > 0)
        assert(MAX_PRICE > 0)
        assert(STEPS_PER_ROUND > 0)
        

def shuffle_quality():
        return min(max(int(np.random.normal(50, 20)), 1), MAX_QUALITY)

