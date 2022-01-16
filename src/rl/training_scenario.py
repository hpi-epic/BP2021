import torch

import agents.vendors as vendors
import market.sim_market as sim_market
import rl.training as training


def run_training_session(marketplace=sim_market.CircularEconomyRebuyPriceOneCompetitor, agent=vendors.QLearningCERebuyAgent):
	"""
	Run a training session with the passed marketplace and QLearningAgent.

	Args:
		marketplace (SimMarket subclass, optional): What marketplace to run the training session on. Defaults to sim_market.CircularEconomyRebuyPriceOneCompetitor.
		agent (QLearningAgent subclass, optional): What kind of QLearningAgent to train. Defaults to vendors.QLearningCERebuyAgent.
	"""
	assert issubclass(marketplace, sim_market.SimMarket), f'the economy passed must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, vendors.QLearningAgent), f'the RL_agent_class passed must be a subclass of QLearningAgent: {agent}'
	assert issubclass(agent, vendors.CircularAgent) == issubclass(marketplace, sim_market.CircularEconomy), 'the agent and marketplace must be of the same economy type (Linear/Circular)'

	marketplace = marketplace()

	n_actions = 1
	if isinstance(marketplace, sim_market.CircularEconomy):
		for id in range(len(marketplace.action_space)):
			n_actions *= marketplace.action_space[id].n
	else:
		n_actions = marketplace.action_space.n

	RL_agent = agent(n_observation=marketplace.observation_space.shape[0], n_actions=n_actions, optim=torch.optim.Adam)
	training.train_QLearning_agent(RL_agent, marketplace)


if __name__ == '__main__':
	run_training_session(sim_market.ClassicScenario, vendors.QLearningLEAgent)
