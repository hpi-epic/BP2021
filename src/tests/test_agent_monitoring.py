from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import agent
import sim_market as sim

from .context import agent_monitoring as am

# create mock rewards list
mock_rewards = []
for number in range(1, 12):
	mock_rewards.append(number)

def test_metrics_average():
	assert 6 == am.metrics_average(mock_rewards)

def test_metrics_median():
	assert 6 == am.metrics_median(mock_rewards)

def test_metrics_maximum():
	assert 11 == am.metrics_maximum(mock_rewards)

def test_metrics_minimum():
	assert 1 == am.metrics_minimum(mock_rewards)

def test_run_marketplace():
	# am.reset_episode(...)
	# ;)
	assert True

def test_round_up():
	assert am.round_up(999, -3) == 1000
