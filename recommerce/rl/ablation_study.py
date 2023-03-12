# This is the script describing the ablation study for the paper.
# It does not contain new framework features, but it stays in the repo to keep the experiments reproducible.

import time
from multiprocessing import Pipe, Process

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO


def run_training_session(market_class, config_market, agent_class, config_rl, training_steps, number, pipe_to_parent):
    agent = agent_class(config_market, config_rl, market_class(config_market, support_continuous_action_space=True), name=f'Train{number}')
    watcher = agent.train_agent(training_steps)
    pipe_to_parent.send(watcher)


def run_group(market_configs, market_descriptions, training_steps, target_function=run_training_session):
    market_class = CircularEconomyRebuyPriceDuopoly
    rl_config = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
    pipes = []
    for _ in market_configs:
        pipes.append(Pipe(False))
    processes = [Process(target=target_function,
                         args=(market_class, config_market, StableBaselinesPPO, rl_config, training_steps, description, pipe_entry))
        for config_market, description, (_, pipe_entry) in zip(market_configs, market_descriptions, pipes)]
    print('Now I start the processes')
    for p in processes:
        time.sleep(10)
        p.start()
    print('Now I wait for the results')
    watchers = [output.recv() for output, _ in pipes]
    print('Now I have the results')
    for p in processes:
        p.join()
    print('All threads joined')
    return market_descriptions, [watcher.get_progress_values_of_property('profits/all', 0) for watcher in watchers]


def get_different_market_configs(parameter_name, values):
    market_configs = []
    descriptions = [f'{parameter_name}={value}' for value in values]
    for value in values:
        market_config = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
        market_config[parameter_name] = value
        market_configs.append(market_config)
    return market_configs, descriptions


if __name__ == '__main__':
    results = run_group(*get_different_market_configs('max_storage', [20, 50, 100, 200]), training_steps=100000)
    print(results)
