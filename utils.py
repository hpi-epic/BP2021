import numpy as np
import json
MAX_QUALITY = 100
MAX_PRICE = 30
PRODUCTION_PRICE = MAX_PRICE / 3
STEPS_PER_ROUND = 50

def setup():
        global STEPS_PER_ROUND, MAX_QUALITY, MAX_PRICE, PRODUCTION_PRICE
        config = {}
        
        with open('config.json') as config_file:
                config = json.load(config_file)
        
        assert('episodes' in config)
        assert('steps_per_round' in config)
        assert('max_quality' in config)
        assert('max_price' in config)
        assert('production_price' in config)
        
        STEPS_PER_ROUND = int(config['episodes'])
        MAX_QUALITY = int(config['max_quality'])
        MAX_PRICE = int(config['max_price'])

        if config['production_price'] == 0:
                PRODUCTION_PRICE = int(MAX_PRICE / 3)
        else:
                PRODUCTION_PRICE = int(config['production_price'])
        
        assert(PRODUCTION_PRICE <= MAX_PRICE)
        assert(PRODUCTION_PRICE > 0)
        assert(MAX_QUALITY > 0)
        assert(MAX_PRICE > 0)
        assert(STEPS_PER_ROUND > 0)
        

def shuffle_quality():
        return min(max(int(np.random.normal(50, 20)), 1), MAX_QUALITY)

