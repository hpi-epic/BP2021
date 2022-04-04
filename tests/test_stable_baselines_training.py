import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.rl.stable_baselines_model as sb_model


@pytest.mark.training
@pytest.mark.slow
def test_ddpg_training():
	sb_model.StableBaselinesDDPG(circular_market.CircularEconomyRebuyPriceOneCompetitor(True), 2).train_agent(120)


@pytest.mark.training
@pytest.mark.slow
def test_td3_training():
	sb_model.StableBaselinesTD3(circular_market.CircularEconomyRebuyPriceOneCompetitor(True), 2).train_agent(120)


@pytest.mark.training
@pytest.mark.slow
def test_a2c_training():
	sb_model.StableBaselinesA2C(circular_market.CircularEconomyRebuyPriceOneCompetitor(True), 2).train_agent(120)


@pytest.mark.training
@pytest.mark.slow
def test_ppo_training():
	sb_model.StableBaselinesPPO(circular_market.CircularEconomyRebuyPriceOneCompetitor(True), 2).train_agent(120)


@pytest.mark.training
@pytest.mark.slow
def test_sac_training():
	sb_model.StableBaselinesSAC(circular_market.CircularEconomyRebuyPriceOneCompetitor(True), 2).train_agent(120)
