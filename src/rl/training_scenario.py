import torch

import agents.vendors as vendors
import market.circular.circular_sim_market as circular_market
import market.linear.linear_sim_market as linear_market
import market.sim_market as sim_market
import rl.training as training


def run_training_session(marketplace=circular_market.CircularEconomyRebuyPriceOneCompetitor, agent=vendors.QLearningCERebuyAgent):
	"""
	Run a training session with the passed marketplace and QLearningAgent.

	Args:
		marketplace (SimMarket subclass, optional): What marketplace to run the training session on.
		Defaults to circular_market.CircularEconomyRebuyPriceOneCompetitor.
		agent (QLearningAgent subclass, optional): What kind of QLearningAgent to train. Defaults to vendors.QLearningCERebuyAgent.
	"""
	assert issubclass(marketplace, sim_market.SimMarket), f'the economy passed must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, vendors.QLearningAgent), f'the RL_agent_class passed must be a subclass of QLearningAgent: {agent}'
	assert issubclass(agent, vendors.CircularAgent) == (issubclass(marketplace, circular_market.CircularEconomy),
		'the agent and marketplace must be of the same economy type (Linear/Circular)')

	marketplace = marketplace()

	agent = agent(n_observation=marketplace.observation_space.shape[0], n_actions=marketplace.get_n_actions(), optim=torch.optim.Adam)
	training.RLTrainer(marketplace, agent).train_QLearning_agent()


if __name__ == '__main__':
	run_training_session(linear_market.ClassicScenario, vendors.QLearningLEAgent)
