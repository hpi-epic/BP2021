# import pytest

# from .context import sim_market, training
# from .context import utils as ut

# This test is currently not working ON THE CI, we believe the cause to be pytorch attempting multithreading
# which is probably not possible/allowed on the CI and/or VM. We will investigate further
# @pytest.mark.parametrize('environment', [sim_market.ClassicScenario(), sim_market.MultiCompetitorScenario(), sim_market.CircularEconomy()])
# def test_market_scenario(environment):
#     ut.REPLAY_START_SIZE = 500
#     training.train_QLearning_agent(environment, int(ut.REPLAY_START_SIZE * 2))
