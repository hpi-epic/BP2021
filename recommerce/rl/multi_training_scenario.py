import time
from multiprocessing import Pipe, Process

import matplotlib.pyplot as plt
import numpy as np

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.self_play import train_self_play
# from recommerce.rl.stable_baselines.sb_td3 import StableBaselinesTD3
from recommerce.rl.stable_baselines.sb_a2c import StableBaselinesA2C
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC


def run_training_session(agent_class, config_rl, number, pipe_to_parent):
    config_market = HyperparameterConfigLoader.load('market_config')
    agent = agent_class(config_market, config_rl, CircularEconomyRebuyPriceDuopoly(config_market,
        support_continuous_action_space=True), name=f'Training_{number}')
    watcher = agent.train_agent(200000)
    pipe_to_parent.send(watcher)


def run_self_play_session(agent_class, config_rl, number, pipe_to_parent):
    watcher = train_self_play(HyperparameterConfigLoader.load('market_config'), config_rl, agent_class, 400000 if issubclass(agent_class, StableBaselinesPPO) else 100000, name=f'SelfPlay_{number}')
    pipe_to_parent.send(watcher)
    # 1000000 if issubclass(agent_class, StableBaselinesPPO) else 200000
    # 400000 if issubclass(agent_class, StableBaselinesPPO) else 100000


def experiment_best_learning_rate_ppo():
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


def experiment_clipping_ppo():
    # Use default parametrisation at all other points
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config'))
    for i, config in enumerate(configs):
        config.clip_range = i * 0.025 + 0.15
        descriptions.append(f'clip_{config["clip_range"]}')

    return configs, descriptions


def experiment_replay_size_sac():
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


def experiment_temperature_sac():
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


def experiment_learning_rate_ddpg():
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ddpg_config'))
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for config, learning_rate in zip(configs, learning_rates):
        config.learning_rate = learning_rate
        descriptions.append(f'ddpg_lr_{learning_rate}')

    return configs, descriptions


def experiment_learning_rate_td3():
    configs = []
    descriptions = []
    for _ in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_td3_config'))
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for config, learning_rate in zip(configs, learning_rates):
        config.learning_rate = learning_rate
        descriptions.append(f'td3_lr_{learning_rate}')

    return configs, descriptions


def experiment_ppo_standard_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config'))
        descriptions.append(f'ppo_standard_{i}')

    return configs, descriptions


def experiment_ppo_clip_0_3_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_ppo_config'))
        configs[-1].clip_range = 0.3
        descriptions.append(f'ppo_clip_range_{0.3}_{i}')

    return configs, descriptions


def experiment_a2c_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_a2c_config'))
        # configs[-1].learning_rate = learning_rate
        descriptions.append(f'a2c_standard_{i}')

    return configs, descriptions


def experiment_sac_all_same():
    configs = []
    descriptions = []
    for i in range(8):
        configs.append(HyperparameterConfigLoader.load('sb_sac_config'))
        # configs[-1].learning_rate = learning_rate
        descriptions.append(f'sac_standard_{i}')

    return configs, descriptions


def run_group(agent, experiment):
    configs, descriptions = experiment()
    print(configs)
    pipes = []
    for _ in configs:
        pipes.append(Pipe(False))
    processes = [Process(target=run_self_play_session, args=(agent, config, description, pipe_entry))
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


if __name__ == '__main__':
    # run_training_session(StableBaselinesSAC, HyperparameterConfigLoader.load('sb_sac_config'), 0)

    # groups = [run_group(StableBaselinesPPO, experiment_clipping_ppo), run_group(StableBaselinesSAC, experiment_temperature_sac)]
    # groups = [run_group(StableBaselinesTD3, experiment_learning_rate_ddpg), run_group(StableBaselinesTD3, experiment_learning_rate_td3)]
    # groups = [
    #     run_group(StableBaselinesA2C, experiment_a2c_all_same),
    #     run_group(StableBaselinesPPO, experiment_ppo_standard_all_same),
    #     run_group(StableBaselinesPPO, experiment_ppo_clip_0_3_all_same)
    # ]
    groups = [run_group(StableBaselinesA2C, experiment_a2c_all_same), run_group(StableBaselinesSAC, experiment_sac_all_same), run_group(StableBaselinesPPO, experiment_ppo_clip_0_3_all_same)]
    # for descriptions, profits_vendor_0 in groups:
    #     print('Next group:')
    #     for descrition, profits in zip(descriptions, profits_vendor_0):
    #         print(f'{descrition} has max of learning curve: {np.max(profits)}')
    #         plt.plot(profits, label=descrition)
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
    plt.title('Comparison of the learning curves')
    plt.xlabel('Episodes')
    plt.ylabel('Profit')
    plt.savefig('multi_training_results.svg')
    plt.show()
