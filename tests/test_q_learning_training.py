import pytest
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.q_learning.q_learning_training as q_learning_training
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

test_scenarios = [
	linear_market.LinearEconomyDuopoly,
	linear_market.LinearEconomyOligopoly,
	circular_market.CircularEconomyMonopoly,
	circular_market.CircularEconomyRebuyPriceMonopoly,
	circular_market.CircularEconomyRebuyPriceDuopoly
]


@pytest.mark.training
@pytest.mark.slow
@pytest.mark.parametrize('marketplace_class', test_scenarios)
def test_market_scenario(marketplace_class):
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceMonopoly)
	config_rl: AttrDict = HyperparameterConfigLoader.load('q_learning_config', QLearningAgent)
	config_rl.replay_start_size = 500
	config_rl.sync_target_frames = 100
	q_learning_training.QLearningTrainer(
		marketplace_class=marketplace_class,
		agent_class=QLearningAgent,
		config_market=config_market,
		config_rl=config_rl
		).train_agent(600)
