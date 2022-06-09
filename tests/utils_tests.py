# import json
# from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
import json
from typing import Tuple, Union

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market


def load_json(path: str):
	"""
	Load a json file.

	Args:
		path (str): The path to the json file.

	Returns:
		dict: The json file as a dictionary.
	"""
	with open(path) as file:
		return json.load(file)


def replace_field_in_dict(initial_dict: dict, key: str, value: Union[str, int, float]) -> dict:
	"""
	Replace a field in a dictionary with a new value.

	Args:
		initial_dict (dict): The dictionary in which to replace the field.
		key (str): The key of the field to be replaced.
		value (Union[str, int, float]): The new value of the field.

	Returns:
		dict: The dictionary with the field replaced.
	"""
	initial_dict[key] = value
	return initial_dict


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
