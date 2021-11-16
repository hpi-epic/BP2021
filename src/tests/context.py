import os
import sys

# inserts the current pyth to sys.path to allow for importing from src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import competitor
# import any classes needed in tests here and call them as such:
# from .context import Customer
import utils
from customer import CustomerLinear
from sim_market import SimMarket
