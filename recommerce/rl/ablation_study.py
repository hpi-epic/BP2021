# This is the script describing the ablation study for the paper.
# It does not contain new framework features, but it stays in the repo to keep the experiments reproducible.

import os
import time
from multiprocessing import Pipe, Process

import numpy as np
import pandas as pd

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly, CircularEconomyRebuyPriceDuopolyFitted
from recommerce.monitoring.exampleprinter import ExamplePrinter
from recommerce.rl.stable_baselines.sb_ppo import StableBaselinesPPO


def create_relevant_dataframe(descriptions, info_sequences_list):
    parameters = [
        ('profit', lambda info_sequence: np.mean(info_sequence['profits/all/vendor_0'])),
        ('new sales', lambda info_sequence: np.mean(info_sequence['customer/purchases_new/vendor_0'])),
        ('refurbished sales', lambda info_sequence: np.mean(info_sequence['customer/purchases_refurbished/vendor_0'])),
        ('rebuys', lambda info_sequence: np.mean(info_sequence['owner/rebuys/vendor_0'])),
        ('offer price new', lambda info_sequence: np.mean(info_sequence['actions/price_new/vendor_0'])),
        ('offer price refurbished', lambda info_sequence: np.mean(info_sequence['actions/price_refurbished/vendor_0'])),
        ('offer price rebuy', lambda info_sequence: np.mean(info_sequence['actions/price_rebuy/vendor_0'])),
        ('sales price new',
            lambda info_sequence: np.sum(np.array(info_sequence['actions/price_new/vendor_0']) *
            np.array(info_sequence['customer/purchases_new/vendor_0'])) /
            np.sum(info_sequence['customer/purchases_new/vendor_0'])),
        ('sales price refurbished',
            lambda info_sequence: np.sum(np.array(info_sequence['actions/price_refurbished/vendor_0']) *
            np.array(info_sequence['customer/purchases_refurbished/vendor_0'])) /
            np.sum(info_sequence['customer/purchases_refurbished/vendor_0'])),
        ('sales price rebuy',
            lambda info_sequence: np.sum(np.array(info_sequence['actions/price_rebuy/vendor_0']) *
            np.array(info_sequence['owner/rebuys/vendor_0'])) /
            np.sum(info_sequence['owner/rebuys/vendor_0'])),
        ('inventory level', lambda info_sequence: np.mean(info_sequence['state/in_storage/vendor_0'])),
        ('profit competitor', lambda info_sequence: np.mean(info_sequence['profits/all/vendor_1'])),
        ('new sales competitor', lambda info_sequence: np.mean(info_sequence['customer/purchases_new/vendor_1'])),
        ('refurbished sales competitor', lambda info_sequence: np.mean(info_sequence['customer/purchases_refurbished/vendor_1'])),
        ('rebuys competitor', lambda info_sequence: np.mean(info_sequence['owner/rebuys/vendor_1'])),
        ('offer price new competitor', lambda info_sequence: np.mean(info_sequence['actions/price_new/vendor_1'])),
        ('offer price refurbished competitor', lambda info_sequence: np.mean(info_sequence['actions/price_refurbished/vendor_1'])),
        ('offer price rebuy competitor', lambda info_sequence: np.mean(info_sequence['actions/price_rebuy/vendor_1'])),
        ('sales price new competitor',
            lambda info_sequence: np.sum(np.array(info_sequence['actions/price_new/vendor_1']) *
            np.array(info_sequence['customer/purchases_new/vendor_1'])) /
            np.sum(info_sequence['customer/purchases_new/vendor_1'])),
        ('sales price refurbished competitor',
            lambda info_sequence: np.sum(np.array(info_sequence['actions/price_refurbished/vendor_1']) *
            np.array(info_sequence['customer/purchases_refurbished/vendor_1'])) /
            (np.sum(info_sequence['customer/purchases_refurbished/vendor_1']) + 1e-10)),
        ('sales price rebuy competitor',
            lambda info_sequence: np.sum(np.array(info_sequence['actions/price_rebuy/vendor_1']) *
            np.array(info_sequence['owner/rebuys/vendor_1'])) /
            (np.sum(info_sequence['owner/rebuys/vendor_1']) + 1e-10)),
        ('inventory level competitor', lambda info_sequence: np.mean(info_sequence['state/in_storage/vendor_1'])),
        ('resources in use', lambda info_sequence: np.mean(info_sequence['state/in_circulation'])),
        ('throw away', lambda info_sequence: np.mean(info_sequence['owner/throw_away']))
    ]

    dataframe_columns = ['market configuration'] + [parameter_name for parameter_name, _ in parameters]

    dataframe = pd.DataFrame(columns=dataframe_columns)
    for description, info_sequences in zip(descriptions, info_sequences_list):
        row = [description]
        for parameter_name, parameter_function in parameters:
            row.append(parameter_function(info_sequences))
        dataframe.loc[len(dataframe)] = row
    return dataframe


def run_training_session(market_class, config_market, agent_class, config_rl, training_steps, number, pipe_to_parent):
    agent = agent_class(config_market, config_rl, market_class(config_market, support_continuous_action_space=True), name=f'Train{number}')
    agent.train_with_default_eval(training_steps)
    exampleprinter = ExamplePrinter(config_market)
    marketplace = market_class(config_market, support_continuous_action_space=True)
    exampleprinter.setup_exampleprinter(marketplace, agent)
    profit, info_sequences = exampleprinter.run_example(save_diagrams=False)
    pipe_to_parent.send(info_sequences)


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
        time.sleep(2)
        p.start()
    print('Now I wait for the results')
    info_sequences = [output.recv() for output, _ in pipes]
    print('Now I have the results')
    for p in processes:
        p.join()
    print('All threads joined')
    return create_relevant_dataframe(market_descriptions, info_sequences)


def get_different_market_configs(parameter_name, values):
    market_configs = []
    descriptions = [f'{parameter_name}={value}' for value in values]
    for value in values:
        market_config = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
        market_config[parameter_name] = value
        market_configs.append(market_config)
    return market_configs, descriptions


if __name__ == '__main__':
    experiments = [('price_step_size', [1.5, 1, 0.5, 0.25])]
    # experiments = [('max_storage', [20, 50, 200]),
    #     ('production_price', [2, 4]),
    #     ('number_of_customers', [10, 30]),
    #     ('storage_cost', [0.01, 0.1, 0.2]),
    #     ('compared_value_old', [0.4, 0.6]),
    #     ('upper_tolerance_old', [4.0, 6.0]),
    #     ('upper_tolerance_new', [7.0, 9.0]),
    #     ('share_interested_owners', [0.025, 0.075]),
    #     ('competitor_lowest_storage_level', [4.5, 8.5]),
    #     ('competitor_ok_storage_level', [9.5, 15.5])
    # ]
    market_configs, descriptions = [], []
    for experiment in experiments:
        print(experiment)
        single_configs, single_descriptions = get_different_market_configs(*experiment)
        market_configs += single_configs
        descriptions += single_descriptions

    print(f'Now I start the experiments. There are {len(market_configs)} experiments in total.')
    dataframes = []
    parallel_runs = 4
    for i in range(0, len(market_configs), parallel_runs):
        print(f'Now I start the experiments {i}-{i+parallel_runs}')
        tmp_dataframe = run_group(market_configs[i:i+parallel_runs], descriptions[i:i+parallel_runs], 1000000)
        dataframes.append(tmp_dataframe)
        print(f'Saving dataframe {i}-{i+parallel_runs}')
        tmp_dataframe.to_excel(os.path.join(PathManager.results_path, f'dataframe{i}-{i+parallel_runs}.xlsx'), index=False)
    dataframe = pd.concat(dataframes)
    print('Now I have the dataframe. I save it...')
    dataframe.to_excel(os.path.join(PathManager.results_path, 'dataframe.xlsx'), index=False)
    print('Done')

    # market_config = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
    # rl_config = HyperparameterConfigLoader.load('sb_ppo_config', StableBaselinesPPO)
    # load_path = os.path.join(PathManager.data_path, 'rl_model_700000_steps.zip')
    # agent = StableBaselinesPPO(market_config, rl_config, CircularEconomyRebuyPriceDuopoly(market_config, support_continuous_action_space=True), name='PPO on fitted market', load_path=load_path)
    # exampleprinter_real = ExamplePrinter(market_config)
    # exampleprinter_real.setup_exampleprinter(CircularEconomyRebuyPriceDuopoly(market_config, support_continuous_action_space=True), agent)
    # _, info_sequences = exampleprinter_real.run_example(save_diagrams=False)
    # exampleprinter_fitted = ExamplePrinter(market_config)
    # exampleprinter_fitted.setup_exampleprinter(CircularEconomyRebuyPriceDuopolyFitted(market_config, support_continuous_action_space=True), agent)
    # _, info_sequences_fitted = exampleprinter_fitted.run_example(save_diagrams=False)
    # dataframe = create_relevant_dataframe(['Values on real market', 'Values on fitted market'], [info_sequences, info_sequences_fitted])
    # dataframe.to_excel(os.path.join(PathManager.results_path, 'dataframe_fitted_vs_real.xlsx'), index=False)
