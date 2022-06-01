import pytest
from attrdict import AttrDict

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceMonopoly
from recommerce.rl.rl_vs_rl_training import train_rl_vs_rl
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO


@pytest.mark.training
@pytest.mark.slow
def test_rl_vs_rl():
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceMonopoly)
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
	train_rl_vs_rl(config_market, config_rl, num_switches=4, num_steps_per_switch=230)
