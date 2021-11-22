import pytest

from .context import training
from .context import sim_market
from .context import utils as ut

@pytest.mark.parametrize('environment', [sim_market.ClassicScenario(), sim_market.MultiCompetitorScenario(), sim_market.CircularEconomy()])
def test_market_scenario(environment):
    ut.REPLAY_START_SIZE = 500
    training.train_QLearning_agent(environment, int(ut.REPLAY_START_SIZE * 2))
