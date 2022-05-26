import pytest
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.stable_baselines.sb_a2c import StableBaselinesA2C
from recommerce.rl.stable_baselines.sb_ddpg import StableBaselinesDDPG
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC
from recommerce.rl.stable_baselines.sb_td3 import StableBaselinesTD3

config_market: AttrDict = HyperparameterConfigLoader.load('market_config')


@pytest.mark.training
@pytest.mark.slow
def test_ddpg_training():
	StableBaselinesDDPG(
		config_market,
		HyperparameterConfigLoader.load('sb_ddpg_config'),
		circular_market.CircularEconomyRebuyPriceDuopoly(
			config=config_market,
			support_continuous_action_space=True)
		).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_td3_training():
	StableBaselinesTD3(
		config_market,
		HyperparameterConfigLoader.load('sb_td3_config'),
		circular_market.CircularEconomyRebuyPriceDuopoly(config=config_market,
		support_continuous_action_space=True)
	).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_a2c_training():
	StableBaselinesA2C(
		config_market,
		HyperparameterConfigLoader.load('sb_a2c_config'),
		circular_market.CircularEconomyRebuyPriceDuopoly(
			config=config_market,
			support_continuous_action_space=True)
		).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_ppo_training():
	StableBaselinesPPO(
		config_market,
		HyperparameterConfigLoader.load('sb_ppo_config'),
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly(
			config=config_market,
			support_continuous_action_space=True)
		).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_sac_training():
	StableBaselinesSAC(
		config_market,
		HyperparameterConfigLoader.load('sb_sac_config'),
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly(
			config=config_market,
			support_continuous_action_space=True)
		).train_agent(1500, 30)
