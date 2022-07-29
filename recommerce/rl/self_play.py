from attrdict import AttrDict

from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


def train_self_play(
		config_market: AttrDict,
		config_rl: AttrDict,
		agent_class: StableBaselinesAgent=StableBaselinesPPO,
		training_steps=1000000,
		name='SelfPlay'):
	tmp_marketplace = CircularEconomyRebuyPriceDuopoly(config=config_market, support_continuous_action_space=True)
	agent: StableBaselinesAgent = agent_class(config_market=config_market, config_rl=config_rl, marketplace=tmp_marketplace, name=name)
	marketplace = CircularEconomyRebuyPriceDuopoly(config=config_market, support_continuous_action_space=True, competitors=[agent])
	agent.set_marketplace(marketplace)

	return agent.train_agent(training_steps=training_steps, iteration_length=50)
