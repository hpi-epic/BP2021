import torch
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.market.sim_market as sim_market
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
import recommerce.rl.q_learning.q_learning_agent as q_learning_agent
import recommerce.rl.rl_vs_rl_training as rl_vs_rl_training
import recommerce.rl.self_play as self_play
from recommerce.configuration.environment_config import EnvironmentConfigLoader, TrainingEnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.market.vendors import FixedPriceAgent
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer
from recommerce.rl.q_learning.q_learning_training import QLearningTrainer
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC

print('successfully imported torch: cuda?', torch.cuda.is_available())


def run_training_session(
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agent=q_learning_agent.QLearningAgent,
		config_market: AttrDict = None,
		config_rl: AttrDict = None,
		competitors: list = None) -> None:
	"""
	Run a training session with the passed marketplace and Agent.

	Args:
		marketplace (SimMarket subclass): What marketplace to run the training session on.
		agent (QLearningAgent subclass): What kind of QLearningAgent to train.
		config_market (AttrDict, optional): The config to be used for the marketplace. Defaults to loading the `market_config`.
		config_rl (AttrDict, optional): The config to be used for the agent. Defaults to loading the `q_learning_config`.
		competitors (list | None, optional): If set, which competitors should be used instead of the default ones.
	"""
	if config_market is None:
		config_market = HyperparameterConfigLoader.load('market_config', marketplace)
	if config_rl is None:
		config_rl = HyperparameterConfigLoader.load('q_learning_config', agent)

	assert issubclass(marketplace, sim_market.SimMarket), \
		f'the type of the passed marketplace must be a subclass of SimMarket: {marketplace}'
	assert issubclass(agent, (q_learning_agent.QLearningAgent, actorcritic_agent.ActorCriticAgent)), \
		f'the RL_agent_class passed must be a subclass of either QLearningAgent or ActorCriticAgent: {agent}'
	if issubclass(marketplace, circular_market.CircularEconomy):
		assert issubclass(agent, CircularAgent), \
			f'The marketplace ({marketplace}) is circular, so all agents need to be circular agents {agent}'

	elif issubclass(marketplace, linear_market.LinearEconomy):
		assert issubclass(agent, LinearAgent), \
			f'The marketplace ({marketplace}) is linear, so all agents need to be linear agents {agent}'

	if issubclass(agent, q_learning_agent.QLearningAgent):
		QLearningTrainer(
			marketplace_class=marketplace,
			agent_class=agent,
			config_market=config_market,
			config_rl=config_rl,
			competitors=competitors).train_agent()
	else:
		ActorCriticTrainer(
			marketplace_class=marketplace,
			agent_class=agent,
			config_rl=config_rl,
			config_market=config_market,
			competitors=competitors
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


def train_continuous_a2c_circular_economy_rebuy():
	"""
	Train an ActorCriticAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	run_training_session(
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agent=actorcritic_agent.ContinuousActorCriticAgentFixedOneStd)


def train_stable_baselines_ppo():
	used_marketplace = circular_market.CircularEconomyRebuyPriceDuopoly
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', used_marketplace)
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
	StableBaselinesPPO(
		config_market=config_market,
		config_rl=config_rl,
		marketplace=used_marketplace(config_market, True)).train_agent()


def train_stable_baselines_sac():
	used_marketplace = circular_market.CircularEconomyRebuyPriceDuopoly
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', used_marketplace)
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', StableBaselinesSAC)
	StableBaselinesSAC(
		config_market=config_market,
		config_rl=config_rl,
		marketplace=used_marketplace(config_market, True)).train_agent()


def train_rl_vs_rl():
	# marketplace is currently hardcoded in train_rl_vs_rl
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceDuopoly)
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
	rl_vs_rl_training.train_rl_vs_rl(config_market, config_rl)


def train_self_play():
	# marketplace is currently hardcoded in train_self_play
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceDuopoly)
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
	self_play.train_self_play(config_market, config_rl)


def train_from_config():
	"""
	Use the `environment_config_training.json` file to decide on the training parameters.
	"""
	config_environment: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	config_rl: AttrDict = HyperparameterConfigLoader.load('q_learning_config', config_environment.agent[0]['agent_class'])
	# TODO: Theoretically, the name of the agent is saved in config['name'], but we don't use it yet.
	competitor_list = []
	for competitor in config_environment.agent[1:]:
		if issubclass(competitor['agent_class'], FixedPriceAgent):
			competitor_list.append(
				competitor['agent_class'](config_market=config_environment, fixed_price=competitor['argument'], name=competitor['name']))
		else:
			competitor_list.append(competitor['agent_class'](config_market=config_environment, name=competitor['name']))

	run_training_session(
		config_rl=config_rl,
		marketplace=config_environment.marketplace,
		agent=config_environment.agent[0]['agent_class'],
		competitors=competitor_list)


def main():
	train_from_config()


if __name__ == '__main__':
	# Make sure a valid datapath is set
	PathManager.manage_user_path()

	main()
