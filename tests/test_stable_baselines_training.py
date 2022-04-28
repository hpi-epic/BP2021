import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.rl.stable_baselines.stable_baselines_model as sb_model
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader, HyperparameterConfig

config_hyperparameter: HyperparameterConfig = HyperparameterConfigLoader.load('hyperparameter_config')


@pytest.mark.training
@pytest.mark.slow
def test_ddpg_training():
	sb_model.StableBaselinesDDPG(config_hyperparameter, circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter, support_continuous_action_space=True)).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_td3_training():
	sb_model.StableBaselinesTD3(config_hyperparameter, circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter, support_continuous_action_space=True)).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_a2c_training():
	sb_model.StableBaselinesA2C(config_hyperparameter, circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter, support_continuous_action_space=True)).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_ppo_training():
	sb_model.StableBaselinesPPO(config_hyperparameter, circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter, support_continuous_action_space=True)).train_agent(1500, 30)


@pytest.mark.training
@pytest.mark.slow
def test_sac_training():
	sb_model.StableBaselinesSAC(config_hyperparameter, circular_market.CircularEconomyRebuyPriceDuopoly(config=config_hyperparameter, support_continuous_action_space=True)).train_agent(1500, 30)
