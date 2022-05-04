import pytest

from recommerce.rl.self_play import train_self_play
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesPPO, StableBaselinesSAC

agents = [StableBaselinesPPO, StableBaselinesSAC]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('agent_class', agents)
def test_self_play(agent_class):
    train_self_play(agent_class, 230)
