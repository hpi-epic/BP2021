
import os

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.monitoring.agent_monitoring.am_monitoring import Monitor
from recommerce.monitoring.consecutive_model_analyzer import analyze_consecutive_models
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO

if __name__ == '__main__':
    config_market = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
    config_rl = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
    monitor = Monitor(config_market, config_rl, 'stable_baselines_ppo_mixed')
    path = 'C:\\Users\\jangr\\OneDrive\\Dokumente\\Bachelorarbeit_Experimente_lokal' + \
        '\\self_play_mixed\\trainedModels\\SelfPlay_ppo_clip_range_0.3_3_Jun14_14-01-45'
    saved_parameter_paths = sorted(
        list(map(lambda x: os.path.join(path, x), filter(lambda filename: 'zip' in filename or 'dat' in filename, os.listdir(path)))))
    analyze_consecutive_models(saved_parameter_paths, monitor, CircularEconomyRebuyPriceDuopoly, config_market, StableBaselinesPPO, True)
