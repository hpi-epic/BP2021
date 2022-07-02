# import math
# import random

# import pandas as pd
# import torch
from attrdict import AttrDict

from recommerce.configuration.environment_config import EnvironmentConfigLoader
# import recommerce.monitoring.exampleprinter as exampleprinter
# import recommerce.rl.training_scenario as training_scenario
# import recommerce.rl.stable_baselines.stable_baselines_model as sbmodel
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgentCompetitive
from recommerce.market.sim_market_kalibrated import SimMarketKalibrated
from recommerce.monitoring.agent_monitoring.am_monitoring import Monitor, run_monitoring_session
# from recommerce.market.sim_market_kalibrated import SimMarketKalibrated
# from recommerce.market_ML.predictable_agent import PredictableAgent
# from recommerce.rl import training_scenario
# from recommerce.rl.stable_baselines import stable_baselines_model as sbmodel
# import recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceDuopoly as CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC


def train_with_calibrated_marketplace_(save_path=None):
	config_environment = EnvironmentConfigLoader.load('environment_config_training')
	config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', config_environment.agent[0]['agent_class'])
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
	model_b = SimMarketKalibrated
	StableBaselinesSAC(config_rl=config_rl, config_market=config_market,
		marketplace=model_b(config_market, True)).train_agent(training_steps=1000, save_path=save_path)


def monitor_agent_model_a():
	monitor = Monitor()
	config_market: AttrDict = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
	# config_rl: AttrDict = HyperparameterConfigLoader.load('sb_sac_config', StableBaselinesSAC)
	marketplace = CircularEconomyRebuyPriceDuopoly
	model_name = 'StableBaselinesSAC_new_training_nn'
	# model_path = f'{model_name}.zip'
	load_path = '/Users/Johann/Documents/GitHub/BP2021/results/trainedModels/StableBaselinesSAC_Jun28_17-13-33/StableBaselinesSAC_00999.zip'
	# os.path.join(PathManager.data_path, model_path)
	# agent = StableBaselinesSAC(config_rl=config_rl, config_market=config_market, load_path=load_path)
	monitor.configurator.setup_monitoring(
		support_continuous_action_space=True,
		episodes=1000,
		plot_interval=200,
		marketplace=marketplace,
		agents=[(StableBaselinesSAC, [load_path, model_name]), (RuleBasedCERebuyAgentCompetitive, ['Rule_based_competitor'])],
		separate_markets=False,
		config_market=config_market
	)
	run_monitoring_session(monitor)


if __name__ == '__main__':
	# Make sure a valid datapath is set
	PathManager.manage_user_path()
	# train_with_pretrained_agent()
	monitor_agent_model_a()
