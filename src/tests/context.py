import os
import sys

# inserts the current pyth to sys.path to allow for importing from src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# import any classes needed in tests here and call them as such:
# from .context import customer
# or: from .context import ClassicScenario
from customer import *
from sim_market import *
import sim_market
from utils import *
import training
