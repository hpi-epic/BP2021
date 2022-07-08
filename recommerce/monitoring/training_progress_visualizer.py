import os

import matplotlib.pyplot as plt
import numpy as np

import recommerce.configuration.utils as ut
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceDuopoly
from recommerce.monitoring.watcher import Watcher


def save_progress_plots(watcher, monitor_path, agent_name, competitors, signature):
	"""
	Save the progress plots of the agent.

	Args:
		watcher (Watcher): The watcher with the progress you want to visualize.
		monitor_path (str): The path the diagrams should be saved at.
		agent_name (str): The name of the agent which was used for the training.
		competitors (list): The competitors during the training.
		signature (str): The signature of the training run.
	"""
	os.makedirs(os.path.join(monitor_path, 'progress_plots'), exist_ok=True)
	os.makedirs(os.path.join(monitor_path, 'scatterplots'), exist_ok=True)
	print('Creating scatterplots...')
	ignore_first_samples = 10  # the number of samples you want to skip because they can be severe outliers
	cumulative_properties = watcher.get_cumulative_properties()
	for property, samples in ut.unroll_dict_with_list(cumulative_properties).items():
		x_values = np.array(range(len(samples[ignore_first_samples:]))) + ignore_first_samples
		plt.clf()

		vendor_index: str = property.rsplit('_', 1)[-1:][0]
		if vendor_index.isdigit():
			# property without '/vendor_x'
			property_name = property.rsplit('/', 1)[:-1][0]
			relevant_agent = agent_name if vendor_index == '0' else competitors[int(vendor_index) - 1].name
			plt.scatter(x_values, samples[ignore_first_samples:], label=relevant_agent, s=10)
			plt.grid()
			plt.title(f'All samples of {property_name}')
			plt.legend()
		else:
			property_name = property
			relevant_agent = None
			plt.scatter(x_values, samples[ignore_first_samples:], s=10)
			plt.grid()
			plt.title(f'All samples of {property_name}')

		plt.xlabel('Episode')
		plt.ylabel(property_name)
		plt.grid(True, linestyle='--')

		if relevant_agent is None:
			filename = f'scatterplot_samples_{property_name.replace("/", "_")}.svg'
		else:
			filename = f'scatterplot_samples_{property_name.replace("/", "_")}_{relevant_agent}.svg'
		plt.savefig(os.path.join(monitor_path, 'scatterplots', filename), transparent=True)

	print('Creating lineplots...')
	for property, samples in cumulative_properties.items():
		plt.clf()
		if isinstance(samples[0], list):
			plot_values = [watcher.get_progress_values_of_property(property, vendor)[ignore_first_samples:]
				for vendor in range(watcher.get_number_of_vendors())]
		else:
			plot_values = [watcher.get_progress_values_of_property(property)[ignore_first_samples:]]

		for vendor_index, values in enumerate(plot_values):
			if vendor_index == 0:
				label = f'{agent_name}'
			else:
				label = competitors[vendor_index - 1].name
			plt.plot(x_values, values, label=label)

		if isinstance(samples[0], list):
			plt.legend()
		plt.title(f'Rolling average training progress of {property}')
		plt.xlabel('Episode')
		plt.ylabel(property)
		plt.grid(True, linestyle='--')
		plt.savefig(os.path.join(monitor_path, 'progress_plots', f'lineplot_progress_{property.replace("/", "_")}.svg'), transparent=True)
		if 'profits' in property:
			bounds = [(0, 12000), (0, 15000), (0, 17500), (0, 10000), (-10000, 10000), (-5000, 10000), (-5000, 5000)]
			for bound in bounds:
				plt.ylim(*bound)
				plt.savefig(
					fname=os.path.join(monitor_path, 'progress_plots',
						f'lineplot_progress_{property.replace("/", "_")}_{bound[0]}_{bound[1]}.svg'),
					transparent=True
				)
		if 'actions' in property:
			plt.ylim(0, 10)
			plt.savefig(
				fname=os.path.join(monitor_path, 'progress_plots',
					f'lineplot_progress_{property.replace("/", "_")}_0_10.svg'),
				transparent=True
			)


def load_and_analyze_existing_watcher_json(path, config_market, agent_name, competitor_names, title):
	"""
	Load a watcher from a json file and analyze it.

	Args:
		path (str): The path to the json file.
	"""
	watcher = Watcher.load_from_json(path, config_market)
	save_progress_plots(watcher, PathManager.results_path, agent_name, competitor_names, title)


if __name__ == '__main__':
	# Load the market config file
	config_market = HyperparameterConfigLoader.load('market_config', CircularEconomyRebuyPriceDuopoly)
	load_and_analyze_existing_watcher_json(
		'C:\\Users\\jangr\\OneDrive\\Dokumente\\Bachelorarbeit_Experimente_lokal\\'
		'comparison_oligopoly\\trainedModels\\Trainsac_standard_1_Jun15_04-53-16\\watchers.json',
		config_market,
		'SAC',
		['Rule Based Undercutting', 'Rule Based (non competitive)', 'Fixed Price', 'Storage Minimizer'],
		'test'
	)
