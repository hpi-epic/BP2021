import os
import shutil
import time
from multiprocessing import Pipe, Process

import matplotlib.pyplot as plt
import numpy as np

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.self_play import train_self_play
from recommerce.rl.stable_baselines.sb_a2c import StableBaselinesA2C
from recommerce.rl.stable_baselines.sb_ddpg import StableBaselinesDDPG
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC
from recommerce.rl.stable_baselines.sb_td3 import StableBaselinesTD3


def run_training_session(agent_class, config_rl, number, pipe_to_parent):
    config_market = HyperparameterConfigLoader.load('market_config')
    agent = agent_class(config_market, config_rl, CircularEconomyRebuyPriceDuopoly(config_market,
        support_continuous_action_space=True), name=f'Training_{number}')
    watcher = agent.train_agent(20000)
    pipe_to_parent.send(watcher)


def run_self_play_session(agent_class, config_rl, number, pipe_to_parent):
    watcher = train_self_play(
        HyperparameterConfigLoader.load('market_config'),
        config_rl, agent_class,
        400000 if issubclass(agent_class, StableBaselinesPPO) else 100000, name=f'SelfPlay_{number}'
    )
    pipe_to_parent.send(watcher)
    # 1000000 if issubclass(agent_class, StableBaselinesPPO) else 200000
    # 400000 if issubclass(agent_class, StableBaselinesPPO) else 100000


def configuration_best_learning_rate_ppo():
    # Use default parametrisation at all other points
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config'))
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
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config'))
    for i, config in enumerate(configs):
        config.clip_range = i * 0.025 + 0.15
        descriptions.append(f'clip_{config["clip_range"]}')

    return configs, descriptions


def configuration_replay_size_sac():
    # Use default parametrisation at all other points
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_sac_config'))
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
        configs.append(HyperparameterConfigLoader.load('sb_sac_config'))
    entropy_coefficient_values = [0.2, 0.5, 1, 1.75, 2.5, 4, 'auto', 'auto']
    for config, entropy_coefficient in zip(configs, entropy_coefficient_values):
        config.ent_coef = entropy_coefficient
        descriptions.append(f'ent_coef_{entropy_coefficient}')

    return configs, descriptions


def configuration_learning_rate_ddpg():
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ddpg_config'))
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for config, learning_rate in zip(configs, learning_rates):
        config.learning_rate = learning_rate
        descriptions.append(f'ddpg_lr_{learning_rate}')

    return configs, descriptions


def configuration_learning_rate_td3():
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_td3_config'))
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for config, learning_rate in zip(configs, learning_rates):
        config.learning_rate = learning_rate
        descriptions.append(f'td3_lr_{learning_rate}')

    return configs, descriptions


def configuration_ppo_standard_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config'))
        descriptions.append(f'ppo_standard_{i}')

    return configs, descriptions


def configuration_ppo_clip_0_3_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config'))
        configs[-1].clip_range = 0.3
        descriptions.append(f'ppo_clip_range_{0.3}_{i}')

    return configs, descriptions


def configuration_a2c_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_a2c_config'))
        # configs[-1].learning_rate = learning_rate
        descriptions.append(f'a2c_standard_{i}')

    return configs, descriptions


def configuration_sac_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_sac_config'))
        # configs[-1].learning_rate = learning_rate
        descriptions.append(f'sac_standard_{i}')

    return configs, descriptions


def run_group(agent, configuration):
    configs, descriptions = configuration()
    print(configs)
    pipes = []
    for _ in configs:
        pipes.append(Pipe(False))
    processes = [Process(target=run_training_session, args=(agent, config, description, pipe_entry))
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
            plt.plot(means, label=f'mean_{descriptions[0][:-2]}')
    plt.legend()
    plt.ylim(0, 1000)
    plt.xlabel('Episodes')
    plt.ylabel('Profit')
    plt.savefig(os.path.join(PathManager.results_path, 'monitoring', f'{name}.svg'))


def experiment_a2c_vs_ppo():
    tasks = [
        (StableBaselinesA2C, configuration_a2c_all_same),
        (StableBaselinesPPO, configuration_ppo_standard_all_same),
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same)
    ]
    groups = [run_group(agent, configuration) for agent, configuration in tasks]
    print_diagrams(groups, 'a2c_ppo')


# move the Path manager results folder to documents
def move_results_to_documents(dest_folder_name):
    folder = shutil.move(
        PathManager.results_path, f'C:\\Users\\jangr\\OneDrive\\Dokumente\\Bachelorarbeit_Experimente\\{dest_folder_name}')
    print(f'Moved results to {folder}')


if __name__ == '__main__':
    groups = [(StableBaselinesPPO, configuration_clipping_ppo), (StableBaselinesSAC, configuration_temperature_sac)]
    groups = [(StableBaselinesDDPG, configuration_learning_rate_ddpg), (StableBaselinesTD3, configuration_learning_rate_td3)]
    groups = [(StableBaselinesA2C, configuration_a2c_all_same), (StableBaselinesSAC, configuration_sac_all_same),
        (StableBaselinesPPO, configuration_ppo_clip_0_3_all_same)]
    print(groups)
    experiment_a2c_vs_ppo()
    move_results_to_documents('ppo_vs_a2c')
