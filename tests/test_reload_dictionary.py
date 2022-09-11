import os

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgentCompetitive
from recommerce.monitoring.training_progress_visualizer import load_and_analyze_existing_watcher_json


def test_reload_and_visualize_watcher():
    config_market = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
    load_and_analyze_existing_watcher_json(
        os.path.join(PathManager.user_path, 'watcher_a2c_run.json'),
        config_market,
        'a2c',
        [RuleBasedCERebuyAgentCompetitive(config_market=config_market, name='Rule Based Undercutting')],
        'a2c_study'
    )
