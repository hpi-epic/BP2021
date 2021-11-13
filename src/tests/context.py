import os
import sys

# inserts the current pyth to sys.path to allow for importing from src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import any classes needed in tests here and call them as such:
# from .context import Customer
from customer import Customer
from sim_market import SimMarket
