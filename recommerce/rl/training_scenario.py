import torch
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.market.sim_market as sim_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
import recommerce.rl.q_learning.q_learning_agent as q_learning_agent
import recommerce.rl.rl_vs_rl_training as rl_vs_rl_training
import recommerce.rl.self_play as self_play
import recommerce.rl.stable_baselines.stable_baselines_model as sbmodel
from recommerce.configuration.environment_config import EnvironmentConfigLoader, TrainingEnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer
from recommerce.rl.q_learning.q_learning_training import QLearningTrainer

print('successfully imported torch: cuda?', torch.cuda.is_available())


def run_training_session(
		config_market: AttrDict = HyperparameterConfigLoader.load('market_config'),
		config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config'),
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agent=q_learning_agent.QLearningAgent):
	"""
	Run a training session with the passed marketplace and QLearningAgent.
	Args:
		marketplace (SimMarket subclass, optional): What marketplace to run the training session on.
		Defaults to circular_market.CircularEconomyRebuyPriceDuopoly.
		agent (QLearningAgent subclass, optional): What kind of QLearningAgent to train. Defaults to q_learning_agent.QLearningAgent.
	"""
	assert issubclass(marketplace, sim_market.SimMarket), \
		f'the type of the passed marketplace must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, (q_learning_agent.QLearningAgent, actorcritic_agent.ActorCriticAgent)), \
		f'the RL_agent_class passed must be a subclass of either QLearningAgent or ActorCriticAgent: {agent}'
	if issubclass(marketplace, circular_market.CircularEconomy):
		assert issubclass(agent, CircularAgent), \
			f'The marketplace({marketplace}) is circular, so all agents need to be circular agehts {agent}'

	elif issubclass(marketplace, linear_market.LinearEconomy):
		assert issubclass(agent, LinearAgent), \
			f'The marketplace({marketplace}) is circular, so all agents need to be circular agehts {agent}'

	if issubclass(agent, q_learning_agent.QLearningAgent):
		QLearningTrainer(
			marketplace_class=marketplace,
			agent_class=agent,
			config_market=config_market,
			config_rl=config_rl).train_agent()
	else:
		ActorCriticTrainer(
			marketplace_class=marketplace,
			agent_class=agent,
			config_rl=config_rl,
			config_market=config_market,
			).train_agent(number_of_training_steps=10000)


# Just add some standard usecases.
def train_q_learning_classic_scenario():
	"""
	Train a Linear QLearningAgent on a Linear Market with one competitor.
	"""
	run_training_session(
		marketplace=linear_market.LinearEconomyDuopoly,
		agent=q_learning_agent.QLearningAgent)


def train_q_learning_circular_economy_rebuy():
	"""
	Train a Circular Economy QLearningAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agent=q_learning_agent.QLearningAgent)


def train_continuos_a2c_circular_economy_rebuy():
	"""
	Train an ActorCriticAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agent=actorcritic_agent.ContinuosActorCriticAgentFixedOneStd)


def train_stable_baselines_ppo():
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
	config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config')
	sbmodel.StableBaselinesPPO(
		config_market=config_market,
		config_rl=config_rl,
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly(config_market, True)).train_agent()


def train_stable_baselines_sac():
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
	config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config')
	sbmodel.StableBaselinesSAC(
		config_market=config_market,
		config_rl=config_rl,
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly(config_market, True)).train_agent()


def train_rl_vs_rl():
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config')
	config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config')
	rl_vs_rl_training.train_rl_vs_rl(config_market, config_rl)


def train_self_play():
	self_play.train_self_play()


def train_from_config():
	"""
	Use the `environment_config_training.json` file to decide on the training parameters.
	"""
	config: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config')
	# TODO: Theoretically, the name of the agent is saved in config['name'], but we don't use it yet.
	run_training_session(
		config_rl=config_rl,
		marketplace=config.marketplace,
		agent=config.agent['agent_class'])


def main():
	train_from_config()


if __name__ == '__main__':
	# Make sure a valid datapath is set
	PathManager.manage_user_path()

	train_rl_vs_rl()
