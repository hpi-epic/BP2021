import os
from typing import Tuple, Union

import market.circular.circular_sim_market as circular_market
import market.linear.linear_sim_market as linear_market


def create_hyperparameter_mock_json_rl(gamma='0.99',
	batch_size='32',
	replay_size='100000',
	learning_rate='1e-6',
	sync_target_frames='1000',
	replay_start_size='10000',
	epsilon_decay_last_frame='75000',
	epsilon_start='1.0',
	epsilon_final='0.1') -> str:
	"""
	Create a string in JSON format that can be used to mock the config_rl.json file.

	Args:
		gamma (str, optional): Defaults to '0.99'.
		batch_size (str, optional): Defaults to '32'.
		replay_size (str, optional): Defaults to '100000'.
		learning_rate (str, optional): Defaults to '1e-6'.
		sync_target_frames (str, optional): Defaults to '1000'.
		replay_start_size (str, optional): Defaults to '10000'.
		epsilon_decay_last_frame (str, optional): Defaults to '75000'.
		epsilon_start (str, optional): Defaults to '1.0'.
		epsilon_final (str, optional): Defaults to '0.1'.

	Returns:
		str: A string in JSON format.
	"""
	return '{\n\t\t"gamma": ' + gamma + ',\n' + \
		'\t\t"batch_size": ' + batch_size + ',\n' + \
		'\t\t"replay_size": ' + replay_size + ',\n' + \
		'\t\t"learning_rate": ' + learning_rate + ',\n' + \
		'\t\t"sync_target_frames": ' + sync_target_frames + ',\n' + \
		'\t\t"replay_start_size": ' + replay_start_size + ',\n' + \
		'\t\t"epsilon_decay_last_frame": ' + epsilon_decay_last_frame + ',\n' + \
		'\t\t"epsilon_start": ' + epsilon_start + ',\n' + \
		'\t\t"epsilon_final": ' + epsilon_final + '\n' + \
		'\t}'


def create_hyperparameter_mock_json_sim_market(
	max_storage='20',
	episode_length='20',
	max_price='15',
	max_quality='100',
	number_of_customers='30',
	production_price='5',
	storage_cost_per_product='0.3') -> str:
	"""
	Create a string in JSON format that can be used to mock the config_sim_market.json file.

	Args:
		max_storage (str, optional): Defaults to '20'.
		episode_length (str, optional): Defaults to '20'.
		max_price (str, optional): Defaults to '15'.
		max_quality (str, optional): Defaults to '100'.
		number_of_customers (str, optional): Defaults to '30'.
		production_price (str, optional): Defaults to '5'.
		storage_cost_per_product (str, optional): Defaults to '0.3'.

	Returns:
		str: A string in JSON format.
	"""
	return '{\n\t\t"max_storage": ' + max_storage + ',\n' + \
		'\t\t"episode_length": ' + episode_length + ',\n' + \
		'\t\t"max_price": ' + max_price + ',\n' + \
		'\t\t"max_quality": ' + max_quality + ',\n' + \
		'\t\t"number_of_customers": ' + number_of_customers + ',\n' + \
		'\t\t"production_price": ' + production_price + ',\n' + \
		'\t\t"storage_cost_per_product": ' + storage_cost_per_product + '\n' + \
		'\t}'


def create_hyperparameter_mock_json(rl: str = create_hyperparameter_mock_json_rl(),
	sim_market: str = create_hyperparameter_mock_json_sim_market()) -> str:
	"""
	Create a mock json in the format of the hyperparameter_config.json.

	Args:
		rl (str, optional): The string that should be used for the rl-part. Defaults to create_hyperparameter_mock_json_rl().
		sim_market (str, optional): The string that should be used for the sim_market-part.
			Defaults to create_hyperparameter_mock_json_sim_market().

	Returns:
		str: The mock json.
	"""
	return '{\n' + '\t"rl": ' + rl + ',\n' + '\t"sim_market": ' + sim_market + '\n}'


def create_environment_mock_dict(
	task: str = 'agent_monitoring',
	enable_live_draw: bool = False,
	episodes: int = 10,
	plot_interval: int = 5,
	marketplace: str = 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	agents: dict = None) -> dict:
	"""
	Create a mock dictionary in the format of an environment_config.json.

	Args:
		task (str, optional): What task to run. Defaults to 'agent_monitoring'.
		enable_live_draw (bool, optional): If live drawing should be enabled. Defaults to False.
		episodes (int, optional): How many episodes to run. Defaults to 10.
		plot_interval (int, optional): How often plots should be drawn. Defaults to 5.
		marketplace (str, optional): What marketplace to run on.
			Defaults to "market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario".
		agents (dict, optional): What agents to use.
			Defaults to {"Fixed CE Rebuy Agent": {"class": "market.vendors.FixedPriceCERebuyAgent"}}.

	Returns:
		dict: The mock dictionary.
	"""
	if agents is None:
		agents = {
			'Fixed CE Rebuy Agent': {
				'class': 'market.vendors.FixedPriceCERebuyAgent'
			}
		}

	return {
		'task': task,
		'enable_live_draw': enable_live_draw,
		'episodes': episodes,
		'plot_interval': plot_interval,
		'marketplace': marketplace,
		'agents': agents
	}


def check_mock_file(mock_file, json) -> None:
	"""
	Confirm that a mock JSON for the config.json is read correctly.

	Args:
		mock_file (unittest.mock.MagicMock): The mocked file.
		json (str): The mock JSON string to be checked.
	"""
	path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'config.json')
	assert open(path).read() == json, 'the mock did not work correctly, as the read file was not equal to the set mock-json'
	mock_file.assert_called_with(path)


def remove_line(number, json) -> str:
	"""
	Remove the specified line from a mock JSON string.

	Args:
		number (int): The line that should be removed.
		json (str): The JSON string from which to remove the line.

	Returns:
		str: The JSON string with the missing line.
	"""
	lines = json.split('\n')
	final_lines = lines[:number + 1]
	final_lines += lines[number + 2:]
	final_lines[-2] = final_lines[-2].replace(',', '')
	return '\n'.join(final_lines)


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
