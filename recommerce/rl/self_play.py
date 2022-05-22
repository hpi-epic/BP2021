from attrdict import AttrDict

from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent, StableBaselinesPPO


def train_self_play(
		config_market: AttrDict,
		config_rl: AttrDict,
		agent_class: StableBaselinesAgent=StableBaselinesPPO,
		training_steps=1000000):
	tmp_marketplace = CircularEconomyRebuyPriceDuopoly(config=config_market, support_continuous_action_space=True)
	agent = agent_class(config_market=config_market, config_rl=config_rl, marketplace=tmp_marketplace)
	marketplace = CircularEconomyRebuyPriceDuopoly(config=config_market, support_continuous_action_space=True, competitors=[agent])
	agent.set_marketplace(marketplace)

	agent.train_agent(training_steps=training_steps)
