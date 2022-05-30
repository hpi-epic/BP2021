import pytest
from attrdict import AttrDict

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.self_play import train_self_play
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC

agents = [(StableBaselinesPPO, 'sb_ppo_config'), (StableBaselinesSAC, 'sb_sac_config')]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('agent_class, config_file', agents)
def test_self_play(agent_class, config_file):
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
	config_rl: AttrDict = HyperparameterConfigLoader.load(config_file)
	train_self_play(config_market=config_market, config_rl=config_rl, agent_class=agent_class, training_steps=230)
