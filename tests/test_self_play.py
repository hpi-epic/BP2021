import pytest
import tests.utils_tests as ut_t
from attrdict import AttrDict

from recommerce.rl.self_play import train_self_play
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesPPO, StableBaselinesSAC

config_hyperparameter: AttrDict = ut_t.mock_config_hyperparameter()
agents = [StableBaselinesPPO, StableBaselinesSAC]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('agent_class', agents)
def test_self_play(agent_class):
    train_self_play(config=config_hyperparameter, agent_class=agent_class, training_steps=230)
