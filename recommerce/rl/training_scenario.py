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
from recommerce.market.sim_market_kalibrated import SimMarketKalibrated
from recommerce.market.vendors import FixedPriceAgent
from recommerce.market_ML.datagenerator_kalibrated_market import KalibratedDatagenerator
from recommerce.market_ML.datagenerator_sim_market import CircularEconomyDatagenerator
from recommerce.market_ML.training_comparer import CircularEconomyComparerMarket
from recommerce.rl.actorcritic.actorcritic_training import ActorCriticTrainer
from recommerce.rl.q_learning.q_learning_training import QLearningTrainer
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent
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
	assert issubclass(agent, (ReinforcementLearningAgent)), \
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
		agent=q_learning_agent.QLearningAgent,)


def train_continuous_ac_circular_economy_rebuy():
	"""
	Train an ActorCriticAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	used_agent = actorcritic_agent.ContinuousActorCriticAgentFixedOneStd
	run_training_session(
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agent=used_agent,
		config_rl=HyperparameterConfigLoader.load('actor_critic_config', used_agent))


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
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', config_environment.agent[0]['agent_class'])
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)

	competitor_list = []
	for competitor in config_environment.agent[1:]:
		if issubclass(competitor['agent_class'], FixedPriceAgent):
			competitor_list.append(
				competitor['agent_class'](config_market=config_market, fixed_price=competitor['argument'], name=competitor['name']))
		else:
			competitor_list.append(competitor['agent_class'](config_market=config_market, name=competitor['name']))

	run_training_session(
		config_rl=config_rl,
		config_market=config_market,
		marketplace=config_environment.marketplace,
		agent=config_environment.agent[0]['agent_class'],
		competitors=competitor_list)


def train_to_calibrate_marketplace():
	"""
	Train an ActorCriticAgent on a Circular Economy Market with Rebuy Prices and one competitor.
	"""
	config_environment: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', config_environment.agent[0]['agent_class'])
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
	run_training_session(
		config_market=config_market,
		config_rl=config_rl,
		marketplace=CircularEconomyDatagenerator,
		agent=actorcritic_agent.ContinuosActorCriticAgentEstimatingStd)


def train_with_calibrated_marketplace(marketplace, save_path=None):
	config_environment: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', config_environment.agent[0]['agent_class'])
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
	StableBaselinesSAC(config_rl=config_rl, config_market=config_market, marketplace=marketplace).train_agent(training_steps=1000,
		save_path=save_path)


def train_with_pretrained_agent(load_path=None):
	if load_path is None:
		load_path = \
			'/Users/Johann/Documents/GitHub/BP2021/results/trainedModels/Stable_Baselines_SAC_May25_12-36-10/Stable_Baselines_SAC_00500.zip'
	config_environment: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', config_environment.agent[0]['agent_class'])
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
	agent = StableBaselinesSAC(
		config_rl=config_rl,
		config_market=config_market,
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly(config_market=config_market, support_continuous_action_space=True),
		load_path=load_path)
	agent.train_agent()


def train_comparer_stable_baselines_sac():
	used_marketplace = CircularEconomyComparerMarket
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', used_marketplace)
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', StableBaselinesSAC)
	StableBaselinesSAC(
		config_market=config_market,
		config_rl=config_rl,
		marketplace=used_marketplace(config_market, True)).train_agent()


def train_with_calibrated_marketplace_(save_path=None):
	config_environment: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', config_environment.agent[0]['agent_class'])
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
	model_b = SimMarketKalibrated
	StableBaselinesSAC(config_rl=config_rl, config_market=config_market,
		marketplace=model_b(config_market, True)).train_agent(training_steps=50000, save_path=save_path)


def data_with_calibrated_marketplace_(save_path=None):
	config_environment: TrainingEnvironmentConfig = EnvironmentConfigLoader.load('environment_config_training')
	# config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', config_environment.agent[0]['agent_class'])
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
	# model_b = KalibratedDatagenerator
	# StableBaselinesSAC(config_rl=config_rl, config_market=config_market,
	# 	marketplace=model_b(config_market, True)).train_agent(training_steps=50000, save_path=save_path)
	model_b = KalibratedDatagenerator(config_market, True)
	for i in range(10):
		print(model_b.step([4, 7, 3])[0])
		model_b.reset()


if __name__ == '__main__':
	# Make sure a valid datapath is set
	PathManager.manage_user_path()
	# train_with_pretrained_agent()
	data_with_calibrated_marketplace_()
