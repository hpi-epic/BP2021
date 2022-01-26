import agents.vendors as vendors
# import market.linear.linear_sim_market as linear_market
import market.circular.circular_sim_market as circular_market
import market.sim_market as sim_market
import rl.actorcritic_agent as actorcritic_agent
from rl.actorcritic_training import ActorCriticTrainer
from rl.q_learning_training import QLearningTrainer


def run_training_session(marketplace=circular_market.CircularEconomyRebuyPriceOneCompetitor, agent=vendors.QLearningCERebuyAgent):
	"""
	Run a training session with the passed marketplace and QLearningAgent.

	Args:
		marketplace (SimMarket subclass, optional): What marketplace to run the training session on.
		Defaults to circular_market.CircularEconomyRebuyPriceOneCompetitor.
		agent (QLearningAgent subclass, optional): What kind of QLearningAgent to train. Defaults to vendors.QLearningCERebuyAgent.
	"""
	assert issubclass(marketplace, sim_market.SimMarket), f'the economy passed must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, vendors.QLearningAgent) or issubclass(agent, actorcritic_agent.ActorCriticAgent), \
		f'the RL_agent_class passed must be a subclass of QLearningAgent: {agent}'
	assert issubclass(agent, vendors.CircularAgent) == issubclass(marketplace, circular_market.CircularEconomy), \
		'the agent and marketplace must be of the same economy type (Linear/Circular)'

	if issubclass(agent, vendors.QLearningAgent):
		QLearningTrainer(marketplace, agent).train_agent()
	else:
		ActorCriticTrainer(marketplace, agent).train_agent(number_of_training_steps=10000)


# Just add some standard usecases.
def q_learning_circular_economy_rebuy():
	run_training_session(circular_market.CircularEconomyRebuyPriceOneCompetitor, vendors.QLearningCERebuyAgent)


def continuos_a2c_circular_economy_rebuy():
	run_training_session(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd)


if __name__ == '__main__':
	continuos_a2c_circular_economy_rebuy()
