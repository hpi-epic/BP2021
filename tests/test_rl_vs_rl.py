import pytest
from attrdict import AttrDict

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.rl_vs_rl_training import train_rl_vs_rl


@pytest.mark.training
@pytest.mark.slow
def test_rl_vs_rl():
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
	config_rl1: AttrDict = HyperparameterConfigLoader.load('sb_ppo_config')
	config_rl2: AttrDict = HyperparameterConfigLoader.load('sb_sac_config')
	train_rl_vs_rl(config_market, config_rl1, config_rl2, num_switches=4, num_steps_per_switch=230)
