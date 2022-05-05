import torch

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.market.sim_market as sim_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
import recommerce.rl.q_learning.q_learning_agent as q_learning_agent
import recommerce.rl.rl_vs_rl_training as rl_vs_rl_training
import recommerce.rl.self_play as self_play
import recommerce.rl.stable_baselines.stable_baselines_model as sbmodel
from recommerce.configuration.environment_config import EnvironmentConfigLoader, TrainingEnvironmentConfig
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.vendors import FixedPriceAgent
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer
from recommerce.rl.q_learning.q_learning_training import QLearningTrainer

print('successfully imported torch: cuda?', torch.cuda.is_available())


def run_training_session(
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agent=q_learning_agent.QLearningAgent,
		competitors: list = None) -> None:
	"""
	Run a training session with the passed marketplace and QLearningAgent.
	Args:
		marketplace (SimMarket subclass): What marketplace to run the training session on.
		agent (QLearningAgent subclass): What kind of QLearningAgent to train.
		competitors (list | None, optional): If set, which competitors should be used instead of the default ones.
	"""
	assert issubclass(marketplace, sim_market.SimMarket), \
		f'the type of the passed marketplace must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, (q_learning_agent.QLearningAgent, actorcritic_agent.ActorCriticAgent)), \
		f'the RL_agent_class passed must be a subclass of either QLearningAgent or ActorCriticAgent: {agent}'
	assert issubclass(agent, CircularAgent) == issubclass(marketplace, circular_market.CircularEconomy), \
		f'the agent and marketplace must be of the same economy type (Linear/Circular): {agent} and {marketplace}'

	if issubclass(agent, q_learning_agent.QLearningAgent):
		QLearningTrainer(marketplace, agent, competitors).train_agent()
	else:
		ActorCriticTrainer(marketplace, agent, competitors).train_agent(number_of_training_steps=10000)


# Just add some standard usecases.
def train_q_learning_classic_scenario():
	"""
	Train a Linear QLearningAgent on a Linear Market with one competitor.
	"""
	run_training_session(linear_market.LinearEconomyDuopoly, q_learning_agent.QLearningAgent)


def train_q_learning_circular_economy_rebuy():
	"""
	Train a Circular Economy QLearningAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(circular_market.CircularEconomyRebuyPriceDuopoly, q_learning_agent.QLearningAgent)


def train_continuos_a2c_circular_economy_rebuy():
	"""
	Train an ActorCriticAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(circular_market.CircularEconomyRebuyPriceDuopoly, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd)


def train_stable_baselines_ppo():
	# For stablebaselines, we do not currently use the given competitors.
	# This should be implemented together with the config for stablebaselines
	sbmodel.StableBaselinesPPO(circular_market.CircularEconomyRebuyPriceDuopoly(True)).train_agent()


def train_stable_baselines_sac():
	sbmodel.StableBaselinesSAC(circular_market.CircularEconomyRebuyPriceDuopoly(True)).train_agent()


def train_rl_vs_rl():
	rl_vs_rl_training.train_rl_vs_rl()


def train_self_play():
	self_play.train_self_play()


def train_from_config():
	"""
	Use the `environment_config_training.json` file to decide on the training parameters.
	"""
	config: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	# TODO: Theoretically, the name of the agent is saved in config['name'], but we don't use it yet.
	# TODO: Same for the competitors, we extract
	competitor_list = []
	for competitor in config.agent[1:]:
		if issubclass(competitor['agent_class'], FixedPriceAgent):
			competitor_list.append(competitor['agent_class'](fixed_price=competitor['argument'], name=competitor['name']))
		else:
			competitor_list.append(competitor['agent_class'](name=competitor['name']))
	run_training_session(config.marketplace, config.agent[0]['agent_class'], competitor_list)


def main():
	train_from_config()


if __name__ == '__main__':
	# Make sure a valid datapath is set
	PathManager.manage_user_path()

	main()
