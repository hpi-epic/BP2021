import json
from typing import Tuple, Union
from unittest.mock import mock_open, patch

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.configuration.hyperparameter_config import HyperparameterConfig, HyperparameterConfigLoader


def create_hyperparameter_mock_dict_rl(gamma: float = 0.99,
	batch_size: int = 32,
	replay_size: int = 500,
	learning_rate: float = 1e-6,
	sync_target_frames: int = 10,
	replay_start_size: int = 100,
	epsilon_decay_last_frame: int = 400,
	epsilon_start: float = 1.0,
	epsilon_final: float = 0.1) -> dict:
	"""
	Create dictionary that can be used to mock the rl part of the hyperparameter_config.json file by calling json.dumps() on it.

	Args:
		gamma (float, optional): Defaults to 0.99.
		batch_size (int, optional): Defaults to 32.
		replay_size (int, optional): Defaults to 100000.
		learning_rate (float, optional): Defaults to 1e-6.
		sync_target_frames (int, optional): Defaults to 1000.
		replay_start_size (int, optional): Defaults to 10000.
		epsilon_decay_last_frame (int, optional): Defaults to 75000.
		epsilon_start (float, optional): Defaults to 1.0.
		epsilon_final (float, optional): Defaults to 0.1.

	Returns:
		dict: The mock dictionary.
	"""
	return {
		'gamma': gamma,
		'batch_size': batch_size,
		'replay_size': replay_size,
		'learning_rate': learning_rate,
		'sync_target_frames': sync_target_frames,
		'replay_start_size': replay_start_size,
		'epsilon_decay_last_frame': epsilon_decay_last_frame,
		'epsilon_start': epsilon_start,
		'epsilon_final': epsilon_final,
	}


def create_hyperparameter_mock_dict_sim_market(
	max_storage: int = 100,
	episode_length: int = 25,
	max_price: int = 10,
	max_quality: int = 50,
	number_of_customers: int = 10,
	production_price: int = 3,
	storage_cost_per_product: float = 0.1) -> dict:
	"""
	Create dictionary that can be used to mock the sim_market part of the hyperparameter_config.json file by calling json.dumps() on it.

	Args:
		max_storage (int, optional): Defaults to 20.
		episode_length (int, optional): Defaults to 20.
		max_price (int, optional): Defaults to 15.
		max_quality (int, optional): Defaults to 100.
		number_of_customers (int, optional): Defaults to 30.
		production_price (int, optional): Defaults to 5.
		storage_cost_per_product (float, optional): Defaults to 0.3.

	Returns:
		dict: The mock dictionary.
	"""
	return {
		'max_storage': max_storage,
		'episode_length': episode_length,
		'max_price': max_price,
		'max_quality': max_quality,
		'number_of_customers': number_of_customers,
		'production_price': production_price,
		'storage_cost_per_product': storage_cost_per_product,
	}


def create_hyperparameter_mock_dict(rl: dict = create_hyperparameter_mock_dict_rl(),
	sim_market: dict = create_hyperparameter_mock_dict_sim_market()) -> dict:
	"""
	Create a dictionary in the format of the hyperparameter_config.json.
	Call json.dumps() on the return value of this to mock the json file.

	Args:
		rl (dict, optional): The dictionary that should be used for the rl-part. Defaults to create_hyperparameter_mock_dict_rl().
		sim_market (dict, optional): The dictionary that should be used for the sim_market-part.
			Defaults to create_hyperparameter_mock_dict_sim_market().

	Returns:
		dict: The mock dictionary.
	"""
	return {
		'rl': rl,
		'sim_market': sim_market
	}


def create_environment_mock_dict(task: str = 'agent_monitoring',
	enable_live_draw: bool = False,
	episodes: int = 10,
	plot_interval: int = 5,
	marketplace: str = 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopoly',
	agents: dict = None) -> dict:
	"""
	Create a mock dictionary in the format of an environment_config.json.

	Args:
		task (str, optional): What task to run. Defaults to 'agent_monitoring'.
		enable_live_draw (bool, optional): If live drawing should be enabled. Defaults to False.
		episodes (int, optional): How many episodes to run. Defaults to 10.
		plot_interval (int, optional): How often plots should be drawn. Defaults to 5.
		marketplace (str, optional): What marketplace to run on.
			Defaults to "recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopoly".
		agents (dict, optional): What agents to use. Defaults to
			[{'name': 'Fixed CE Rebuy Agent', 'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent', 'argument': ''}].

	Returns:
		dict: The mock dictionary.
	"""
	if agents is None:
		agents = [
			{
				'name': 'Fixed CE Rebuy Agent',
				'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
				'argument': ''
			}
		]

	return {
		'task': task,
		'enable_live_draw': enable_live_draw,
		'episodes': episodes,
		'plot_interval': plot_interval,
		'marketplace': marketplace,
		'agents': agents
	}


def create_combined_mock_dict(hyperparameter: dict or None = create_hyperparameter_mock_dict(),
	environment: dict or None = create_environment_mock_dict()) -> dict:
	"""
	Create a mock dictionary in the format of a configuration file with both a hyperparameter and environment part.
	If any of the two parameters is `None`, leave that key out of the resulting dictionary.

	Args:
		hyperparameter (dict | None, optional): The hyperparameter part of the combined config. Defaults to create_hyperparameter_mock_dict().
		environment (dict | None, optional): The environment part of the combined config. Defaults to create_environment_mock_dict().

	Returns:
		dict: The mock dictionary.
	"""
	if hyperparameter is None and environment is None:
		return {}
	elif hyperparameter is None:
		return {
			'environment': environment
		}
	elif environment is None:
		return {
			'hyperparameter': hyperparameter
		}
	return {
		'hyperparameter': hyperparameter,
		'environment': environment
	}


def check_mock_file(mock_file, mocked_file_content) -> None:
	"""
	Confirm that a mock JSON is read correctly.

	Args:
		mock_file (unittest.mock.MagicMock): The mocked file.
		mocked_file_content (str): The mocked_file_content to be checked.
	"""
	path = 'some_path'
	with open(path) as file:
		assert file.read() == mocked_file_content, \
			'the mock did not work correctly, as the read file was not equal to the set mocked_file_content'
	mock_file.assert_called_with(path)


def remove_key(key: str, original_dict: dict) -> dict:
	"""
	Remove the specified key from a dictionary and return the dictionary.

	Args:
		key (str): The key that should be removed.
		json (dict): The dictionary from which to remove the line.

	Returns:
		dict: The dictionary without the key.
	"""
	original_dict.pop(key)
	return original_dict


def create_mock_rewards(num_entries) -> list:
	"""
	Create a list of ints to be used as e.g. mock rewards.

	Args:
		num_entries (int): How many numbers should be in the list going from 1 to num_entries.

	Returns:
		list: The list of rewards.
	"""
	return list(range(1, num_entries))


def create_mock_action(market_subclass) -> Union[int, Tuple]:
	"""
	Create an array to be used as an action. The length of the array fits to the argument's class.

	Args:
		market_subclass (SimMarket): A non-abstract subclass for which an action will be returned.

	Returns:
		list: An action array with mocked values.
	"""
	if issubclass(market_subclass, linear_market.LinearEconomy):
		return 1
	elif issubclass(market_subclass, circular_market.CircularEconomyRebuyPrice):
		return (1, 2, 3)
	elif issubclass(market_subclass, circular_market.CircularEconomy):
		return (1, 2)


def mock_config_hyperparameter() -> HyperparameterConfig:
	"""
	Reload the hyperparameter_config file to update the config variable with the mocked values.

	Returns:
		HyperparameterConfig: The mocked hyperparameter config object.
	"""
	mock_json = json.dumps(create_hyperparameter_mock_dict())
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		check_mock_file(mock_file, mock_json)
		config_hyperparameter = HyperparameterConfigLoader.load('hyperparameter_config')
		return config_hyperparameter
