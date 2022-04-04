import pytest

import recommerce.market.circular.circular_sim_market as circular_market
from recommerce.rl.stable_baselines_training import train_ddpg_agent


@pytest.mark.training
@pytest.mark.slow
def test_ddpg_training():
	train_ddpg_agent(circular_market.CircularEconomyRebuyPriceOneCompetitor, 120, 2)
