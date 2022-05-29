import time
from multiprocessing import Process

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.sb_sac import StableBaselinesSAC


def run_training_session(agent_class, config_rl, number):
    config_market = HyperparameterConfigLoader.load('market_config')
    agent = agent_class(config_market, config_rl, CircularEconomyRebuyPriceDuopoly(config_market,
        support_continuous_action_space=True), name=f'SAC_{number}')
    agent.train_agent(200000)  # (400000)


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
        config.clip_range = i * 0.025 + 0.1
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
    entropy_coefficient_values = [0.1, 0.2, 0.5, 1, 1.75, 2.5, 4, 7]
    for config, entropy_coefficient in zip(configs, entropy_coefficient_values):
        config.ent_coef = entropy_coefficient
        descriptions.append(f'ent_coef_{entropy_coefficient}')

    return configs, descriptions


if __name__ == '__main__':
    # run_training_session(StableBaselinesSAC, HyperparameterConfigLoader.load('sb_sac_config'), 0)
    configs, descriptions = experiment_temperature_sac()
    print(configs)
    processes = [Process(target=run_training_session, args=(StableBaselinesSAC, config, description))
        for config, description in zip(configs, descriptions)]
    print('Now I start the processes')
    for p in processes:
        time.sleep(5)
        p.start()
    print('Now I wait for the results')
    for p in processes:
        p.join()
    print('Here you are')
