import copy
import json
import os
import signal
import sys
import time
from json import JSONEncoder

import matplotlib.pyplot as plt
import numpy as np
import torch
from attrdict import AttrDict
from torch.utils.tensorboard import SummaryWriter

import recommerce.configuration.utils as ut
import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.configuration.environment_config import EnvironmentConfigLoader, ExampleprinterEnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent
from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import Agent, FixedPriceAgent
from recommerce.monitoring.svg_manipulation import SVGManipulator
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)

class ExamplePrinter():

    def __init__(self, config_market: AttrDict):
        ut.ensure_results_folders_exist()
        self.config_market = config_market
        self.marketplace = circular_market.CircularEconomyRebuyPriceDuopoly(config=self.config_market)
        self.agent = RuleBasedCERebuyAgent(config_market=self.config_market)
        # Signal handler for e.g. KeyboardInterrupt
        signal.signal(signal.SIGINT, self._signal_handler)

    def setup_exampleprinter(self, marketplace: SimMarket = None, agent: Agent = None) -> None:
        """
        Configure the current exampleprinter session.

        Args:
            marketplace (SimMarket instance, optional): What marketplace to run the session on.
            agent (Agent instance, optional): What agent ot run the session on..
        """
        if (marketplace is not None):
            self.marketplace = marketplace
        if (agent is not None):
            self.agent = agent

    def _signal_handler(self, signum, frame) -> None:  # pragma: no cover
        """
        Handle any interruptions to the running process, such as a `KeyboardInterrupt`-event.
        """
        print('\nAborting exampleprinter run...')
        sys.exit(0)

    def run_example(self, save_lineplots=False) -> int:
        """
        Run a specified marketplace with a (pre-trained, if RL) agent and record various statistics using TensorBoard.

        Args:
            save_lineplots (bool, optional): Whether to save lineplots of the market's performance.

        Returns:
            int: The profit made.
        """
        print(
            f'Running exampleprinter on a {self.marketplace.__class__.__name__} market with a {self.agent.__class__.__name__} agent...')
        counter = 0
        our_profit = 0
        is_done = False
        state = self.marketplace.reset()

        signature = f'exampleprinter_{time.strftime("%b%d_%H-%M-%S")}'
        writer = SummaryWriter(log_dir=os.path.join(PathManager.results_path, 'runs', signature))
        os.makedirs(os.path.join(PathManager.results_path, 'exampleprinter', signature))

        # is circular can be used for line plots (all types: monopoly, duopoly, oligopoly)
        is_circular_rebuy = isinstance(self.marketplace, circular_market.CircularEconomyRebuyPrice)

        # linear and circular can be used for example printer svg
        is_circular_rebuy_doupoly = isinstance(self.marketplace, circular_market.CircularEconomyRebuyPriceDuopoly)
        is_linear_doupoly = isinstance(self.marketplace, linear_market.LinearEconomyDuopoly)

        is_circular = isinstance(self.marketplace, circular_market.CircularEconomy)

        new_price_name = "price_new" if is_circular else "price"
        new_purchases_name = "purchases_new" if is_circular else "purchases"

        if is_circular_rebuy_doupoly or is_linear_doupoly:
            svg_manipulator = SVGManipulator(signature,
                                             isinstance(self.marketplace, linear_market.LinearEconomyDuopoly))

        cumulative_dict = None

        profits = [[] for _ in range(self.marketplace._number_of_vendors)]
        price_news = [[] for _ in range(self.marketplace._number_of_vendors)]
        sales_new = [[] for _ in range(self.marketplace._number_of_vendors)]
        sales_no_buy = []

        if is_circular and save_lineplots:
            price_refurbished = [[] for _ in range(self.marketplace._number_of_vendors)]
            price_rebuy = [[] for _ in range(self.marketplace._number_of_vendors)]
            in_storages = [[] for _ in range(self.marketplace._number_of_vendors)]
            sales_refurbished = [[] for _ in range(self.marketplace._number_of_vendors)]
            in_circulations = []



        with torch.no_grad():
            while not is_done:
                action = self.agent.policy(state)
                print(state)
                print(action)
                state, reward, is_done, logdict = self.marketplace.step(action)
                if cumulative_dict is not None:
                    cumulative_dict = ut.add_content_of_two_dicts(cumulative_dict, logdict)
                else:
                    cumulative_dict = copy.deepcopy(logdict)
                ut.write_dict_to_tensorboard(writer, logdict, counter)
                ut.write_dict_to_tensorboard(writer, cumulative_dict, counter, is_cumulative=True,
                                             episode_length=self.config_market.episode_length)
                if is_circular_rebuy_doupoly or is_linear_doupoly:
                    ut.write_content_of_dict_to_overview_svg(svg_manipulator, counter, logdict, cumulative_dict,
                                                             self.config_market, is_linear_doupoly)

                our_profit += reward
                counter += 1

                for i in range(self.marketplace._number_of_vendors):
                    price_news[i].append(logdict[f"actions/{new_price_name}"][f'vendor_{i}'])
                    sales_new[i].append(logdict[f"customer/{new_purchases_name}"][f'vendor_{i}'])
                    profits[i].append(logdict['profits/all'][f'vendor_{i}'])

                sales_no_buy.append(logdict['customer/buy_nothing'])
                if is_circular_rebuy and save_lineplots:
                    for i in range(self.marketplace._number_of_vendors):
                        price_refurbished[i].append(logdict['actions/price_refurbished'][f'vendor_{i}'])
                        price_rebuy[i].append(logdict['actions/price_rebuy'][f'vendor_{i}'])
                        in_storages[i].append(logdict['state/in_storage'][f'vendor_{i}'])
                        sales_refurbished[i].append(logdict['customer/purchases_refurbished'][f'vendor_{i}'])

                    in_circulations.append(logdict['state/in_circulation'])

                if is_circular_rebuy_doupoly or is_linear_doupoly:
                    svg_manipulator.save_overview_svg(filename=('MarketOverview_%.3d' % counter))


        raw_data = {
            'vendors': self.marketplace._number_of_vendors,
            'is_linear': not is_circular,
            'price_new': price_news,
            'sales_new': sales_new,
            'sales_no_buy': sales_no_buy,
            'profits': profits
        }

        if is_circular:
            raw_data['max_storage'] = self.marketplace.max_storage
            raw_data['price_refurbished'] = price_refurbished  # refurbished verkaufpreis
            raw_data['price_rebuy'] = price_rebuy  # rueckkauf preis!
            raw_data['sales_refurbished'] = sales_refurbished
            raw_data['in_circulation'] = in_circulations
            raw_data['in_storage'] = in_storages


        with open(os.path.join(PathManager.results_path, 'exampleprinter', signature, 'raw_data.json'), 'w') as f:
            json.dump(raw_data, f, cls=NumpyFloatValuesEncoder)




        if is_circular_rebuy_doupoly or is_linear_doupoly:
            svg_manipulator.to_html()

        if is_circular_rebuy and save_lineplots:
            self.save_step_diagrams(price_refurbished, price_news, price_rebuy, in_storages,
                                    in_circulations, sales_refurbished, sales_new, sales_no_buy, signature)

        return our_profit

    def save_step_diagrams(self, price_refurbished, price_news, price_rebuy, in_storages, in_circulations, sales_refurbished,
                           sales_new, sales_no_buy, signature) -> None:
        x = np.array(range(1, self.config_market.episode_length + 1))
        plt.step(x, in_circulations)
        plt.savefig(os.path.join(PathManager.results_path, 'exampleprinter', signature, 'lineplot_in_circulations.svg'))
        plt.xlim(450, 475)
        plt.savefig(
            os.path.join(PathManager.results_path, 'exampleprinter', signature, 'lineplot_in_circulations_xlim.svg'),
            transparent=True)
        plt.clf()

        plt.figure(figsize=(100, 6))
        plt.step(x, sales_no_buy, label='# no buy')

        for i in range(self.marketplace._number_of_vendors):
            plt.step(x - (0.5 if i == 1 else 0),
                     sales_refurbished[i],
                     label=f'# rebuy customer {self.agent.name if i == 0 else self.marketplace.competitors[i - 1].name}')
            plt.step(x - (0.5 if i == 1 else 0),
                     sales_new[i],
                     label=f'# new buy customer {self.agent.name if i == 0 else self.marketplace.competitors[i - 1].name}')
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(PathManager.results_path, 'exampleprinter', signature, 'customer_behaviour.svg'))

        plt.clf()
        plt.figure(figsize=(plt.rcParamsDefault['figure.figsize']))

        for data, name in [(price_refurbished, 'price_refurbished'),
                           (price_news, 'price_new'),
                           (price_rebuy, 'price_rebuy'),
                           (in_storages, 'in_storages'),
                           ]:
            for i in range(self.marketplace._number_of_vendors):
                plt.step(x - (0.5 if i == 1 else 0),
                         data[i], label=(self.agent.name if i == 0 else self.marketplace.competitors[i - 1].name))
            plt.legend()
            plt.title(f'Step Diagram of {name}')
            plt.xlabel('Step')
            plt.ylabel(name)
            if 'price' in name:
                plt.ylim(0, 10)
            elif 'in_storage' in name:
                plt.ylim(0, 100)
            plt.savefig(os.path.join(PathManager.results_path, 'exampleprinter', signature, f'lineplot_{name}.svg'),
                        transparent=True)
            plt.xlim(450, 475)
            plt.savefig(
                os.path.join(PathManager.results_path, 'exampleprinter', signature, f'lineplot_{name}_xlim.svg'),
                transparent=True)
            plt.clf()


def main():  # pragma: no cover
    """
    Defines what is performed when the `agent_monitoring` command is chosen in `main.py`.
    """
    config_environment: ExampleprinterEnvironmentConfig = EnvironmentConfigLoader.load(
        'environment_config_exampleprinter')

    config_market: AttrDict = HyperparameterConfigLoader.load('market_config', config_environment.marketplace)
    config_rl: AttrDict = HyperparameterConfigLoader.load('rl_config', config_environment.agent[0]['agent_class'])
    printer = ExamplePrinter(config_market=config_market)

    competitor_list = []
    for competitor in config_environment.agent[1:]:
        if issubclass(competitor['agent_class'], FixedPriceAgent):
            competitor_list.append(
                competitor['agent_class'](config_market=config_market, fixed_price=competitor['argument'],
                                          name=competitor['name']))
        else:
            competitor_list.append(competitor['agent_class'](config_market=config_market, name=competitor['name']))

    # TODO: Theoretically, the name of the agent is saved in config_environment['name'], but we don't use it yet.
    marketplace = config_environment.marketplace(config=config_market, competitors=competitor_list)

    # QLearningAgents need more initialization
    if issubclass(config_environment.agent[0]['agent_class'], (QLearningAgent, StableBaselinesAgent)):

        agent = config_environment.agent[0]['agent_class'](
            config_market=config_market,
            config_rl=config_rl,
            marketplace=marketplace,
            load_path=os.path.abspath(os.path.join(PathManager.data_path, config_environment.agent[0]['argument'])))

        printer.setup_exampleprinter(
            marketplace=marketplace,
            agent=agent)
    else:
        printer.setup_exampleprinter(marketplace=marketplace, agent=config_environment.agent[0]['agent_class']())

    print(f'The final profit was: {printer.run_example(save_lineplots=True)}')


if __name__ == '__main__':  # pragma: no cover
    # Make sure a valid datapath is set
    PathManager.manage_user_path()

    main()
