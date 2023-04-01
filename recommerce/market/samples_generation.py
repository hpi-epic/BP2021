import os

from tqdm import tqdm

from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgentSSCurve
from recommerce.monitoring.exampleprinter import ExamplePrinter

if __name__ == '__main__':
	config_market = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
	exampleprinter = ExamplePrinter(config_market)
	agent = RuleBasedCERebuyAgentSSCurve(config_market, 'Sample Collector', True)
	marketplace = CircularEconomyRebuyPriceDuopoly(config_market, True, document_for_regression=True)
	exampleprinter.setup_exampleprinter(marketplace, agent)
	for _ in tqdm(range(20)):
		exampleprinter.run_example(False)
	print('Saving customers dataframe...')
	marketplace.customers_dataframe.to_excel(os.path.join(PathManager.results_path, 'customers_dataframe.xlsx'), index=False)
	print('Saving owners dataframe...')
	marketplace.owners_dataframe.to_excel(os.path.join(PathManager.results_path, 'owners_dataframe.xlsx'), index=False)
	print('Saving reaction dataframe...')
	marketplace.competitor_reaction_dataframe.to_excel(
		os.path.join(PathManager.results_path, 'competitor_reaction_dataframe.xlsx'), index=False)
