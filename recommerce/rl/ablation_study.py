# This is the script describing the ablation study for the paper.
# It does not contain new framework features, but it stays in the repo to keep the experiments reproducible.

import os
import time
from multiprocessing import Pipe, Process

import numpy as np
import pandas as pd

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
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
            np.sum(info_sequence['customer/purchases_refurbished/vendor_1'])),
        ('sales price rebuy competitor',
            lambda info_sequence: np.sum(np.array(info_sequence['actions/price_rebuy/vendor_1']) *
            np.array(info_sequence['owner/rebuys/vendor_1'])) /
            np.sum(info_sequence['owner/rebuys/vendor_1'])),
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
    profit, info_sequences = exampleprinter.run_example(save_lineplots=True)
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
        time.sleep(10)
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
    storage_df = run_group(*get_different_market_configs('max_storage', [20, 50, 100, 200]), training_steps=100000)
    print(storage_df)
    storage_df.to_excel(os.path.join(PathManager.results_path, 'storage.xlsx'), index=False)
    production_price_df = run_group(*get_different_market_configs('production_price', [2, 3, 4]), training_steps=100000)
    print(production_price_df)
    production_price_df.to_excel(os.path.join(PathManager.results_path, 'production_price.xlsx'), index=False)
    number_of_customers_df = run_group(*get_different_market_configs('number_of_customers', [10, 20, 30]), training_steps=100000)
    print(number_of_customers_df)
    number_of_customers_df.to_excel(os.path.join(PathManager.results_path, 'number_of_customers.xlsx'), index=False)
    storage_cost_df = run_group(*get_different_market_configs('storage_cost', [0.01, 0.05, 0.1, 0.2]), training_steps=100000)
    print(storage_cost_df)
    storage_cost_df.to_excel(os.path.join(PathManager.results_path, 'storage_cost.xlsx'), index=False)
    compared_value_old_df = run_group(*get_different_market_configs('compared_value_old', [0.4, 0.55, 0.6]), training_steps=100000)
    print(compared_value_old_df)
    compared_value_old_df.to_excel(os.path.join(PathManager.results_path, 'compared_value_old.xlsx'), index=False)
    upper_tolerance_old_df = run_group(*get_different_market_configs('upper_tolerance_old', [4.0, 5.0, 6.0]), training_steps=100000)
    print(upper_tolerance_old_df)
    upper_tolerance_old_df.to_excel(os.path.join(PathManager.results_path, 'upper_tolerance_old.xlsx'), index=False)
    upper_tolerance_new_df = run_group(*get_different_market_configs('upper_tolerance_new', [7.0, 8.0, 9.0]), training_steps=100000)
    print(upper_tolerance_new_df)
    upper_tolerance_new_df.to_excel(os.path.join(PathManager.results_path, 'upper_tolerance_new.xlsx'), index=False)
    share_interested_owners_df = run_group(*get_different_market_configs('share_interested_owners', [0.025, 0.05, 0.075]),
                                           training_steps=100000)
    print(share_interested_owners_df)
    share_interested_owners_df.to_excel(os.path.join(PathManager.results_path, 'share_interested_owners.xlsx'), index=False)
    competitor_lowest_storage_level_df = run_group(*get_different_market_configs('competitor_lowest_storage_level', [4.5, 6.5, 8.5]),
                                                   training_steps=100000)
    print(competitor_lowest_storage_level_df)
    competitor_lowest_storage_level_df.to_excel(os.path.join(PathManager.results_path, 'competitor_lowest_storage_level.xlsx'),
                                                index=False)
    competitor_ok_storage_level_df = run_group(*get_different_market_configs('competitor_ok_storage_level', [9.5, 12.5, 15.5]),
                                               training_steps=100000)

    # merge all dataframes
    all_dataframes = [storage_df, production_price_df, number_of_customers_df, storage_cost_df, compared_value_old_df,
                    upper_tolerance_old_df, upper_tolerance_new_df, share_interested_owners_df,
                    competitor_lowest_storage_level_df, competitor_ok_storage_level_df]
    all_dataframes = [df.set_index('market configuration') for df in all_dataframes]
    merged_df = pd.concat(all_dataframes, axis=1)

    # save merged dataframe to excel
    merged_df.to_excel(os.path.join(PathManager.results_path, 'merged.xlsx'))
