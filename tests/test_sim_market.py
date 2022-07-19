import pytest
import utils_tests as ut_t
from attrdict import AttrDict

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.rl.stable_baselines.sb_a2c import StableBaselinesA2C
from recommerce.rl.stable_baselines.sb_ddpg import StableBaselinesDDPG
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC
from recommerce.rl.stable_baselines.sb_td3 import StableBaselinesTD3

market_classes = [
	linear_market.LinearEconomyDuopoly,
	linear_market.LinearEconomyOligopoly,
	circular_market.CircularEconomyMonopoly,
	circular_market.CircularEconomyRebuyPriceDuopoly,
	circular_market.CircularEconomyRebuyPriceOligopoly,
	circular_market.CircularEconomyRebuyPriceMonopoly,
	circular_market.CircularEconomyRebuyPriceDuopoly,
	circular_market.CircularEconomyRebuyPriceOligopoly
]


market_combinations = []
for market_class in market_classes:
	# In the following loop, we iterate through all combinations boolean hyperparameters.
	# We will test each combination.
	# The three toggles are opposite_own_state_visibility, common_state_visibility and reward_mixed_profit_and_difference
	for i in range(8):
		config_market = HyperparameterConfigLoader.load('market_config', market_class)
		config_market.opposite_own_state_visibility = i % 2
		if issubclass(market_class, linear_market.LinearEconomy) and 1 - i % 2:
			continue
		config_market.common_state_visibility = (i // 2) % 2
		config_market.reward_mixed_profit_and_difference = (i // 4) % 2
		# reward_mixed_profit_and_difference only works for non monopoly markets
		if issubclass(market_class, (circular_market.CircularEconomyMonopoly, circular_market.CircularEconomyRebuyPriceMonopoly)) \
			and (i // 4) % 2:
			continue
		market_combinations.append((market_class, config_market))

agent_classes = [
	(QLearningAgent, 'q_learning_config', False),
	(actorcritic_agent.DiscreteActorCriticAgent, 'actor_critic_config', False),
	(actorcritic_agent.ContinuousActorCriticAgentFixedOneStd, 'actor_critic_config', False),
	(actorcritic_agent.ContinuousActorCriticAgentEstimatingStd, 'actor_critic_config', False),
	(StableBaselinesA2C, 'sb_a2c_config', True),
	(StableBaselinesPPO, 'sb_ppo_config', True),
	(StableBaselinesDDPG, 'sb_ddpg_config', True),
	(StableBaselinesTD3, 'sb_td3_config', True),
	(StableBaselinesSAC, 'sb_sac_config', True)
]


market_initialization_and_steps_testcases = [(*market_related, *agent_related)
	for market_related, agent_related in ut.cartesian_product(market_combinations, agent_classes)]
market_initialization_and_steps_testcases = list(filter(lambda x: not (issubclass(x[2], (StableBaselinesA2C, StableBaselinesPPO)) and
	issubclass(x[0], linear_market.LinearEconomy)), market_initialization_and_steps_testcases))


@pytest.mark.parametrize('marketclass', market_classes)
def test_unique_output_dict(marketclass):
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceMonopoly)
	market = marketclass(config=config_market)
	_, _, _, info_dict_1 = market.step(ut_t.create_mock_action(marketclass))
	_, _, _, info_dict_2 = market.step(ut_t.create_mock_action(marketclass))
	assert id(info_dict_1) != id(info_dict_2)


@pytest.mark.parametrize('market_class, config_market, agent_class, config_agent, continuos_action_space',
	market_initialization_and_steps_testcases)
def test_market_initialization_and_steps(market_class, config_market, agent_class, config_agent, continuos_action_space):
	market = market_class(config_market, support_continuous_action_space=continuos_action_space)
	config_rl = HyperparameterConfigLoader.load(config_agent, agent_class)
	agent = agent_class(config_market=config_market, config_rl=config_rl, marketplace=market, name='test')
	if isinstance(agent, QLearningAgent):
		agent.optimizer = None
	state = market.reset()
	for _ in range(10):
		action = agent.policy(state)
		print(action)
		state, _, _, _ = market.step(action)
