import alpha_business.agents.vendors as vendors
import alpha_business.market.circular.circular_sim_market as circular_market
import alpha_business.market.linear.linear_sim_market as linear_market
import alpha_business.market.sim_market as sim_market
import alpha_business.rl.actorcritic_agent as actorcritic_agent
from alpha_business.configuration.environment_config import EnvironmentConfigLoader, TrainingEnvironmentConfig
from alpha_business.rl.actorcritic_training import ActorCriticTrainer
from alpha_business.rl.q_learning_training import QLearningTrainer


def run_training_session(marketplace=circular_market.CircularEconomyRebuyPriceOneCompetitor, agent=vendors.QLearningCERebuyAgent):
	"""
	Run a training session with the passed marketplace and QLearningAgent.

	Args:
		marketplace (SimMarket subclass, optional): What marketplace to run the training session on.
		Defaults to circular_market.CircularEconomyRebuyPriceOneCompetitor.
		agent (QLearningAgent subclass, optional): What kind of QLearningAgent to train. Defaults to vendors.QLearningCERebuyAgent.
	"""
	assert issubclass(marketplace, sim_market.SimMarket), f'the marketplace passed must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, (vendors.QLearningAgent, actorcritic_agent.ActorCriticAgent)), \
		f'the RL_agent_class passed must be a subclass of either QLearningAgent or ActorCriticAgent: {agent}'
	assert issubclass(agent, vendors.CircularAgent) == issubclass(marketplace, circular_market.CircularEconomy), \
		f'the agent and marketplace must be of the same economy type (Linear/Circular): {agent} and {marketplace}'

	if issubclass(agent, vendors.QLearningAgent):
		QLearningTrainer(marketplace, agent).train_agent()
	else:
		ActorCriticTrainer(marketplace, agent).train_agent(number_of_training_steps=10000)


# Just add some standard usecases.
def train_q_learning_classic_scenario():
	"""
	Train a Linear QLearningAgent on a Linear Market with one competitor.
	"""
	run_training_session(linear_market.ClassicScenario, vendors.QLearningLEAgent)


def train_q_learning_circular_economy_rebuy():
	"""
	Train a Circular Economy QLearningAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(circular_market.CircularEconomyRebuyPriceOneCompetitor, vendors.QLearningCERebuyAgent)


def train_continuos_a2c_circular_economy_rebuy():
	"""
	Train an ActorCriticAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd)


def train_from_config():
	"""
	Use the `environment_config.json` file to decide on the training parameters.
	"""
	config: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	run_training_session(config.marketplace, config.agent)


def main():  # pragma: no cover
	"""
	Defines what is performed when the `agent_monitoring` command is chosen in `main.py`.
	"""
	train_from_config()


if __name__ == '__main__':
	main()
