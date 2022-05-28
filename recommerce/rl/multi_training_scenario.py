import time
from multiprocessing import Process

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO


def run_training_session(agent_class, config_rl, number):
    config_market = HyperparameterConfigLoader.load('market_config')
    agent = agent_class(config_market, config_rl, CircularEconomyRebuyPriceDuopoly(config_market,
        support_continuous_action_space=True), name=f'PPO_{number}')
    agent.train_agent(1000000)


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


if __name__ == '__main__':
    # run_training_session(StableBaselinesPPO, HyperparameterConfigLoader.load("sb_ppo_config"), 0)
    configs, descriptions = experiment_best_learning_rate_ppo()
    processes = [Process(target=run_training_session, args=(StableBaselinesPPO, config, description))
        for config, description in zip(configs, descriptions)]
    print('Now I start the processes')
    for p in processes:
        p.start()
        time.sleep(5)
    print('Now I wait for the results')
    for p in processes:
        p.join()
    print('Here you are')
