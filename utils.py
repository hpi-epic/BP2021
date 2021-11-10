import numpy as np

MAX_QUALITY = 100
MAX_PRICE = 30
PRODUCTION_PRICE = MAX_PRICE / 3

def shuffle_quality():
        return min(max(int(np.random.normal(50, 20)), 1), MAX_QUALITY)