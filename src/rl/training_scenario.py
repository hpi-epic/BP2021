import torch

import agents.vendors as vendors
import market.sim_market as sim_market
import rl.training as training


def run_training_session(marketplace_class=sim_market.CircularEconomyRebuyPriceOneCompetitor, RL_agent_class=vendors.QLearningCERebuyAgent):
	"""
	Run a training session with the passed marketplace and QLearningAgent.

	Args:
		marketplace_class (SimMarket subclass, optional): What marketplace to run the training session on. Defaults to sim_market.CircularEconomyRebuyPriceOneCompetitor.
		RL_agent_class (QLearningAgent subclass, optional): What kind of QLearningAgent to train. Defaults to vendors.QLearningCERebuyAgent.
	"""
	assert issubclass(marketplace_class, sim_market.SimMarket), f'the economy passed must be a subclass of SimMarket: {marketplace_class}'
	assert issubclass(RL_agent_class, vendors.QLearningAgent), f'the RL_agent_class passed must be a subclass of QLearningAgent: {RL_agent_class}'
	assert issubclass(RL_agent_class, vendors.CircularAgent) == issubclass(marketplace_class, sim_market.CircularEconomy), 'the agent and marketplace must be of the same economy type (Linear/Circular)'

	marketplace = marketplace_class()

	n_actions = 1
	if isinstance(marketplace, sim_market.CircularEconomy):
		for id in range(len(marketplace.action_space)):
			n_actions *= marketplace.action_space[id].n

	RL_agent = RL_agent_class(n_observation=marketplace.observation_space.shape[0], n_actions=n_actions, optim=torch.optim.Adam)
	training.train_QLearning_agent(RL_agent, marketplace)


if __name__ == '__main__':
	run_training_session()
