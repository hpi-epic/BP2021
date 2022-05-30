import time
from multiprocessing import Pipe, Process

import matplotlib.pyplot as plt
import numpy as np

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC


def run_training_session(agent_class, config_rl, number, pipe_to_parent):
    config_market = HyperparameterConfigLoader.load('market_config')
    agent = agent_class(config_market, config_rl, CircularEconomyRebuyPriceDuopoly(config_market,
        support_continuous_action_space=True), name=f'Training_{number}')
    watcher = agent.train_agent(500000)
    pipe_to_parent.send(watcher)


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


def run_group(agent, experiment):
    configs, descriptions = experiment()
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


if __name__ == '__main__':
    # run_training_session(StableBaselinesSAC, HyperparameterConfigLoader.load('sb_sac_config'), 0)

    groups = [run_group(StableBaselinesPPO, experiment_clipping_ppo), run_group(StableBaselinesSAC, experiment_temperature_sac)]
    for descriptions, profits_vendor_0 in groups:
        print('Next group:')
        for descrition, profits in zip(descriptions, profits_vendor_0):
            print(f'{descrition} has max of learning curve: {np.max(profits)}')
            plt.plot(profits, label=descrition)
    plt.legend()
    plt.ylim(0, 1000)
    plt.title('Comparison of the learning curves')
    plt.xlabel('Episodes')
    plt.ylabel('Profit')
    plt.savefig('multi_training_results.svg')
    plt.show()
