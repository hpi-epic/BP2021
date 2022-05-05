from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly, CircularEconomyRebuyPriceVariableDuopoly
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent, StableBaselinesPPO


def train_self_play(agent_class: StableBaselinesAgent = StableBaselinesPPO, training_steps=1000000):
	tmp_marketplace = CircularEconomyRebuyPriceDuopoly(True)
	agent = agent_class(tmp_marketplace)
	marketplace = CircularEconomyRebuyPriceVariableDuopoly(agent)
	agent.set_marketplace(marketplace)

	agent.train_agent(training_steps=training_steps)
