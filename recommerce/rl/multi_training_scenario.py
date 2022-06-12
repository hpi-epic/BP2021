import os
import shutil
import time
from multiprocessing import Pipe, Process

import matplotlib.pyplot as plt
import numpy as np

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import (CircularEconomyRebuyPriceDuopoly, CircularEconomyRebuyPriceMonopoly,
                                                            CircularEconomyRebuyPriceOligopoly)
from recommerce.rl.self_play import train_self_play
from recommerce.rl.stable_baselines.sb_a2c import StableBaselinesA2C
from recommerce.rl.stable_baselines.sb_ddpg import StableBaselinesDDPG
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC
from recommerce.rl.stable_baselines.sb_td3 import StableBaselinesTD3


def run_training_session(market_class, config_market_path, agent_class, config_rl, training_steps, number, pipe_to_parent):
    config_market = HyperparameterConfigLoader.load(config_market_path, market_class)
    agent = agent_class(config_market, config_rl, market_class(config_market, support_continuous_action_space=True), name=f'Train{number}')
    watcher = agent.train_agent(training_steps)
    pipe_to_parent.send(watcher)


def run_self_play_session(market_class, config_market_path, agent_class, config_rl, training_steps, number, pipe_to_parent):
    assert issubclass(market_class, CircularEconomyRebuyPriceDuopoly)
    config_market = HyperparameterConfigLoader.load(config_market_path, market_class)
    watcher = train_self_play(config_market, config_rl, agent_class, training_steps, name=f'SelfPlay_{number}')
    pipe_to_parent.send(watcher)


def configuration_best_learning_rate_ppo():
    # Use default parametrisation at all other points
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO))
    for i, config in enumerate(configs):
        config.learning_rate = (i + 1) * 1.5e-5 + 2e-5
        descriptions.append(f'lr_{config["learning_rate"]}')
        print(f'The {i}th one has learning rate {config.learning_rate}')

    return configs, descriptions


def configuration_clipping_ppo():
    # Use default parametrisation at all other points
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO))
    for i, config in enumerate(configs):
        config.clip_range = i * 0.1 / 3 + 0.2
        descriptions.append(f'clip_{config["clip_range"]}')

    return configs, descriptions


def configuration_replay_size_sac():
    # Use default parametrisation at all other points
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_sac_config', StableBaselinesSAC))
    sizes = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    for config, size in zip(configs, sizes):
        config.buffer_size = size
        descriptions.append(f'buffer_size_{size}')

    return configs, descriptions


def configuration_temperature_sac():
    # Use default parametrisation at all other points
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_sac_config', StableBaselinesSAC))
    entropy_coefficient_values = [0.2, 0.5, 1, 1.75, 2.5, 4, 'auto', 'auto']
    for config, entropy_coefficient in zip(configs, entropy_coefficient_values):
        config.ent_coef = entropy_coefficient
        descriptions.append(f'ent_coef_{entropy_coefficient}')

    return configs, descriptions


def configuration_learning_rate_ddpg():
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ddpg_config', StableBaselinesDDPG))
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for config, learning_rate in zip(configs, learning_rates):
        config.learning_rate = learning_rate
        descriptions.append(f'ddpg_lr_{learning_rate}')

    return configs, descriptions


def configuration_learning_rate_td3():
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_td3_config', StableBaselinesTD3))
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for config, learning_rate in zip(configs, learning_rates):
        config.learning_rate = learning_rate
        descriptions.append(f'td3_lr_{learning_rate}')

    return configs, descriptions


def configuration_ppo_standard_all_same():
    configs = []
    descriptions = []
    for i in range(4):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO))
        descriptions.append(f'ppo_standard_{i}')

    return configs, descriptions


def configuration_ppo_clip_0_3_all_same():
    configs = []
    descriptions = []
    for i in range(4):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO))
        configs[-1].clip_range = 0.3
        descriptions.append(f'ppo_clip_range_{0.3}_{i}')

    return configs, descriptions


def configuration_a2c_all_same():
    configs = []
    descriptions = []
    for i in range(4):
        configs.append(HyperparameterConfigLoader.load('sb_a2c_config', StableBaselinesA2C))
        # configs[-1].learning_rate = learning_rate
        descriptions.append(f'a2c_standard_{i}')

    return configs, descriptions


def configuration_sac_all_same():
    configs = []
    descriptions = []
    for i in range(4):
        configs.append(HyperparameterConfigLoader.load('sb_sac_config', StableBaselinesSAC))
        # configs[-1].learning_rate = learning_rate
        descriptions.append(f'sac_standard_{i}')

    return configs, descriptions


def run_group(market_class, config_market, agent, configuration, training_steps, target_function=run_training_session):
    configs, descriptions = configuration()
    print(configs)
    pipes = []
    for _ in configs:
        pipes.append(Pipe(False))
    processes = [Process(target=target_function, args=(market_class, config_market, agent, config, training_steps, description, pipe_entry))
        for config, description, (_, pipe_entry) in zip(configs, descriptions, pipes)]
    print('Now I start the processes')
    for p in processes:
        time.sleep(2)
        p.start()
    print('Now I wait for the results')
    watchers = [output.recv() for output, _ in pipes]
    print('Now I have the results')
    for p in processes:
        p.join()
    print('All threads joined')
    return descriptions, [watcher.get_progress_values_of_property('profits/all', 0) for watcher in watchers]


def print_diagrams(groups, name, individual_lines=False):
    plt.clf()
    plt.figure(figsize=(10, 5))
    if individual_lines:
        for descriptions, profits_vendor_0 in groups:
            print('Next group:')
            for descrition, profits in zip(descriptions, profits_vendor_0):
                print(f'{descrition} has max of learning curve: {np.max(profits)}')
                plt.plot(profits, label=descrition)
    else:
        for descriptions, profits_vendor_0 in groups:
            profits_vendor_0 = np.array(profits_vendor_0)
            print(f'The individual maximums were: {profits_vendor_0.max(axis=1)}')
            mins = np.min(profits_vendor_0, axis=0)
            maxs = np.max(profits_vendor_0, axis=0)
            means = np.mean(profits_vendor_0, axis=0)
            plt.fill_between(range(len(mins)), mins, maxs, alpha=0.5)
            plt.plot(means, label=f'Mean of {descriptions[0][:-2]}')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Profit')
    plt.savefig(os.path.join(PathManager.results_path, 'monitoring', f'{name}.svg'), transparent=True)
    plt.ylim(0, 12000)
    plt.savefig(os.path.join(PathManager.results_path, 'monitoring', f'{name}_clipped_12000.svg'), transparent=True)
    plt.ylim(0, 15000)
    plt.savefig(os.path.join(PathManager.results_path, 'monitoring', f'{name}_clipped_15000.svg'), transparent=True)
    plt.ylim(0, 17500)
    plt.savefig(os.path.join(PathManager.results_path, 'monitoring', f'{name}_clipped_17500.svg'), transparent=True)
    plt.ylim(0, 10000)
    plt.savefig(os.path.join(PathManager.results_path, 'monitoring', f'{name}_clipped_10000.svg'), transparent=True)


# These experiments are all done on a CERebuy market
standardtraining = 1000000  # 1000000
shorttraining = 300000
market_class = CircularEconomyRebuyPriceDuopoly


def experiment_a2c_vs_ppo():
    tasks = [
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesPPO, configuration_ppo_standard_all_same),
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same)
    ]
    groups = [run_group(market_class, 'market_config', agent, configuration, standardtraining) for agent, configuration in tasks]
    print_diagrams(groups, 'a2c_ppo')


def experiment_a2c_vs_sac():
    tasks = [
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesSAC, configuration_sac_all_same)
    ]
    groups = [run_group(market_class, 'market_config', agent, configuration, shorttraining) for agent, configuration in tasks]
    print_diagrams(groups, 'a2c_sac')


def experiment_self_play():
    tasks = [
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same),
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesSAC, configuration_sac_all_same)
    ]
    groups = [run_group(market_class, 'market_config', agent, configuration, standardtraining, target_function=run_self_play_session)
        for agent, configuration in tasks]
    print_diagrams(groups, 'self_play')


def experiment_several_ddpg_td3():
    tasks = [(StableBaselinesDDPG, configuration_learning_rate_ddpg), (StableBaselinesTD3, configuration_learning_rate_td3)]
    groups = [run_group(market_class, 'market_config', agent, configuration, standardtraining) for agent, configuration in tasks]
    print_diagrams(groups, 'ddpg_td3', True)
    print_diagrams([groups[0]], 'ddpg', True)
    print_diagrams([groups[1]], 'td3', True)


def experiment_higher_clip_ranges_ppo():
    tasks = [(StableBaselinesPPO, configuration_clipping_ppo)]
    groups = [run_group(market_class, 'market_config', agent, configuration, standardtraining) for agent, configuration in tasks]
    print_diagrams(groups, 'ppo_clipping', True)


def experiment_temperature_sac():
    tasks = [(StableBaselinesSAC, configuration_temperature_sac)]
    groups = [run_group(market_class, 'market_config', agent, configuration, shorttraining) for agent, configuration in tasks]
    print_diagrams(groups, 'sac_temperature', True)


def experiment_partial_markov_a2c():
    groups = [
        run_group(market_class, 'market_config', StableBaselinesA2C, configuration_a2c_all_same, shorttraining),
        run_group(market_class, 'market_config2', StableBaselinesA2C, configuration_a2c_all_same, shorttraining),
        run_group(market_class, 'market_config3', StableBaselinesA2C, configuration_a2c_all_same, shorttraining)
    ]
    print_diagrams(groups, 'partial_markov_a2c')


def experiment_partial_markov_ppo():
    groups = [
        run_group(market_class, 'market_config', StableBaselinesPPO, configuration_ppo_clip_0_3_all_same, standardtraining),
        run_group(market_class, 'market_config2', StableBaselinesPPO, configuration_ppo_clip_0_3_all_same, standardtraining),
        run_group(market_class, 'market_config3', StableBaselinesPPO, configuration_ppo_clip_0_3_all_same, standardtraining)
    ]
    print_diagrams(groups, 'partial_markov_ppo')


def experiment_partial_markov_sac():
    groups = [
        run_group(market_class, 'market_config', StableBaselinesSAC, configuration_sac_all_same, shorttraining),
        run_group(market_class, 'market_config2', StableBaselinesSAC, configuration_sac_all_same, shorttraining),
        run_group(market_class, 'market_config3', StableBaselinesSAC, configuration_sac_all_same, shorttraining)
    ]
    print_diagrams(groups, 'partial_markov_sac')


def experiment_mixed_rewards_a2c():
    groups = [
        run_group(market_class, 'market_config3', StableBaselinesA2C, configuration_a2c_all_same, shorttraining),
        run_group(market_class, 'market_config_mixed', StableBaselinesA2C, configuration_a2c_all_same, shorttraining)
    ]
    print_diagrams(groups, 'mixed_rewards_a2c')


def experiment_mixed_rewards_ppo():
    groups = [
        run_group(market_class, 'market_config3', StableBaselinesPPO, configuration_ppo_clip_0_3_all_same, standardtraining),
        run_group(market_class, 'market_config_mixed', StableBaselinesPPO, configuration_ppo_clip_0_3_all_same, standardtraining)
    ]
    print_diagrams(groups, 'mixed_rewards_ppo')


def experiment_mixed_rewards_sac():
    groups = [
        run_group(market_class, 'market_config3', StableBaselinesSAC, configuration_sac_all_same, shorttraining),
        run_group(market_class, 'market_config_mixed', StableBaselinesSAC, configuration_sac_all_same, shorttraining)
    ]
    print_diagrams(groups, 'mixed_rewards_sac')


def experiment_self_play_mixed():
    tasks = [
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same),
        (StableBaselinesSAC, configuration_sac_all_same)
    ]
    groups = [run_group(market_class, 'market_config_mixed', agent, configuration, standardtraining, target_function=run_self_play_session)
        for agent, configuration in tasks]
    print_diagrams(groups, 'self_play_mixed')


def experiment_monopoly():
    tasks = [
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same),
        (StableBaselinesSAC, configuration_sac_all_same)
    ]
    groups = [run_group(
        CircularEconomyRebuyPriceMonopoly, 'market_config', agent, configuration, standardtraining, target_function=run_training_session)
        for agent, configuration in tasks]
    print_diagrams(groups, 'comparison_monopoly')


def experiment_oligopol():
    tasks = [
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same),
        (StableBaselinesSAC, configuration_sac_all_same)
    ]
    groups = [run_group(
        CircularEconomyRebuyPriceOligopoly, 'market_config', agent, configuration, standardtraining, target_function=run_training_session)
        for agent, configuration in tasks]
    print_diagrams(groups, 'comparison_oligopoly')


def experiment_oligopol_mixed():
    tasks = [
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same),
        (StableBaselinesSAC, configuration_sac_all_same)
    ]
    groups = [run_group(CircularEconomyRebuyPriceOligopoly,
        'market_config_mixed', agent, configuration, standardtraining, target_function=run_training_session)
        for agent, configuration in tasks]
    print_diagrams(groups, 'comparison_oligopoly_mixed')


# move the Path manager results folder to documents
def move_results_to_documents(dest_folder_name):
    folder = shutil.move(
        PathManager.results_path, f'C:\\Users\\jangr\\OneDrive\\Dokumente\\Bachelorarbeit_Experimente\\{dest_folder_name}')
    print(f'Moved results to {folder}')


if __name__ == '__main__':
    # experiment_a2c_vs_ppo()
    # move_results_to_documents('a2c_vs_ppo')
    # experiment_a2c_vs_sac()
    # move_results_to_documents('a2c_vs_sac')
    # experiment_self_play()
    # move_results_to_documents('self_play')
    # experiment_several_ddpg_td3()
    # move_results_to_documents('ddpg_td3')
    # experiment_higher_clip_ranges_ppo()
    # move_results_to_documents('ppo_clipping')
    # experiment_temperature_sac()
    # move_results_to_documents('sac_temperature')
    # experiment_partial_markov_a2c()
    # move_results_to_documents('partial_markov_a2c')
    # experiment_partial_markov_ppo()
    # move_results_to_documents('partial_markov_ppo')
    # experiment_partial_markov_sac()
    # move_results_to_documents('partial_markov_sac')
    experiment_mixed_rewards_a2c()
    move_results_to_documents('mixed_rewards_a2c')
    experiment_mixed_rewards_ppo()
    move_results_to_documents('mixed_rewards_ppo')
    experiment_mixed_rewards_sac()
    move_results_to_documents('mixed_rewards_sac')
    experiment_self_play_mixed()
    move_results_to_documents('self_play_mixed')
    experiment_monopoly()
    move_results_to_documents('comparison_monopoly')
    experiment_oligopol()
    move_results_to_documents('comparison_oligopoly')
    experiment_oligopol_mixed()
    move_results_to_documents('comparison_oligopoly_mixed')
