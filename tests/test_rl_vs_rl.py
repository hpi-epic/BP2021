import pytest
from attrdict import AttrDict

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.rl_vs_rl_training import train_rl_vs_rl


@pytest.mark.training
@pytest.mark.slow
def test_rl_vs_rl():
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_ppo_config')
	train_rl_vs_rl(config_market, config_rl, num_switches=4, num_steps_per_switch=230)
