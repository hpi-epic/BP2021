import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.market.sim_market as sim_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
import recommerce.rl.q_learning.q_learning_agent as q_learning_agent
from recommerce.configuration.environment_config import EnvironmentConfigLoader, TrainingEnvironmentConfig
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer
from recommerce.rl.q_learning.q_learning_training import QLearningTrainer


def run_training_session(marketplace=circular_market.CircularEconomyRebuyPriceOneCompetitor, agent=q_learning_agent.QLearningCERebuyAgent):
	"""
	Run a training session with the passed marketplace and QLearningAgent.
	Args:
		marketplace (SimMarket subclass, optional): What marketplace to run the training session on.
		Defaults to circular_market.CircularEconomyRebuyPriceOneCompetitor.
		agent (QLearningAgent subclass, optional): What kind of QLearningAgent to train. Defaults to q_learning_agent.QLearningCERebuyAgent.
	"""
	assert issubclass(marketplace, sim_market.SimMarket), f'the marketplace passed must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, (q_learning_agent.QLearningAgent, actorcritic_agent.ActorCriticAgent)), \
		f'the RL_agent_class passed must be a subclass of either QLearningAgent or ActorCriticAgent: {agent}'
	assert issubclass(agent, CircularAgent) == issubclass(marketplace, circular_market.CircularEconomy), \
		f'the agent and marketplace must be of the same economy type (Linear/Circular): {agent} and {marketplace}'

	if issubclass(agent, q_learning_agent.QLearningAgent):
		QLearningTrainer(marketplace, agent).train_agent()
	else:
		ActorCriticTrainer(marketplace, agent).train_agent(number_of_training_steps=10000)


# Just add some standard usecases.
def train_q_learning_classic_scenario():
	"""
	Train a Linear QLearningAgent on a Linear Market with one competitor.
	"""
	run_training_session(linear_market.ClassicScenario, q_learning_agent.QLearningLEAgent)


def train_q_learning_circular_economy_rebuy():
	"""
	Train a Circular Economy QLearningAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(circular_market.CircularEconomyRebuyPriceOneCompetitor, q_learning_agent.QLearningCERebuyAgent)


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


def main():
	train_from_config()


if __name__ == '__main__':
	train_continuos_a2c_circular_economy_rebuy()
