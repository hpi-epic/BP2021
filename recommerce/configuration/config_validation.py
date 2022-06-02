# This file contains logic used by the webserver to validate configuration files

from recommerce.configuration.environment_config import EnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigValidator


def validate_config(config: dict, config_is_final: bool) -> tuple:
	"""
	Validates a given config dictionary either uploaded by the user or entered into the form before starting a container.

	Args:
		config (dict): The config to validate.
		config_is_final (bool): Whether or not the config must contain all required keys.

	Returns:
		tuple: success: A status (True) and the split hyperparameter_config and environment_config dictionaries as a tuple.
				failure: A status (False) and the errormessage as a string.
	"""
	try:
		# we only allow users to upload either a hyperparameter_config (rl or market) or an environment_config, not combined!
		config_type = find_config_type(config)

		# we can only validate types for the environment_config, as we do not know the agent/market class for the rl/market configs
		if config_type == 'environment':
			# validate that all given values have the correct types
			task = config['task'] if config_is_final else 'None'
			EnvironmentConfig.check_types(config, task, False, config_is_final)

		# the webserver needs another format of the config
		config.pop('config_type')
		if config_type == 'rl':
			return True, {'rl': config}
		elif config_type == 'sim_market':
			return True, {'sim_market': config}
		else:
			return True, {'environment': config}

	except Exception as error:
		return False, str(error)


def validate_sub_keys(config_class: HyperparameterConfigValidator or EnvironmentConfig, config: dict, top_level_keys: dict) -> None:
	"""
	Utility function that validates if a given config contains only allowed keys.
	Can be used recursively for dictionaries within dictionaries.
	"Unisex": Works for both HyperparameterConfigValidator and EnvironmentConfig.

	Args:
		config_class (HyperparameterConfigValidator or EnvironmentConfig): The config class from which to get the required fields.
		config (dict): The config given by the user.
		top_level_keys (dict): The keys of the current level. Their values indicate if there is another dictionary expected for that key.

	Raises:
		AssertionError: If the given config contains a key that is invalid.
	"""
	for key, _ in config.items():
		# we need to separately check agents, since it is a list of dictionaries
		if key == 'agents':
			assert isinstance(config['agents'], list), f'The "agents" key must have a value of type list, but was {type(config["agents"])}'
			for agent in config['agents']:
				assert isinstance(agent, dict), f'All agents must be of type dict, but this one was {type(agent)}'
				assert all(agent_key in {'name', 'agent_class', 'argument'} for agent_key in agent.keys()), \
					f'An invalid key for agents was provided: {agent.keys()}'
		# the key is key of a dictionary in the config
		elif top_level_keys[key]:
			assert isinstance(config[key], dict), f'The value of this key must be of type dict: {key}, but was {type(config[key])}'
			# these are the valid keys that sub-key can have as keys in the dictionary
			key_fields = config_class.get_required_fields(key)
			# check that only valid keys were given by the user
			for sub_key, _ in config[key].items():
				assert sub_key in key_fields.keys(), \
					f'The key "{sub_key}" should not exist within a {config_class.__name__} config (was checked at sub-key "{key}")'
			# if there is an additional layer of dictionaries, check it recursively
			validate_sub_keys(config_class, config[key], key_fields)


def find_config_type(config: dict) -> str:
	"""
	Extract the config type from the config dictionary. Config type is defined by the "config_type" key, which must always be present.

	Args:
		config (dict): The config to check.

	Raises:
		AssertionError: If the config_type key has an invalid value or is missing.

	Returns:
		str: The config type.
	"""
	try:
		if config['config_type'] in ['rl', 'market', 'environment']:
			return config['config_type']
		else:
			raise AssertionError(f'the "config_type" key must be one of "rl", "market", "environment" but was {config["config_type"]}')
	except KeyError as e:
		raise AssertionError(f"your config is missing the 'config_type' key, which must be one of 'rl', 'market', 'environment': {config}") from e


def split_mixed_config(config: dict) -> tuple:
	"""
	Utility function that splits a potentially mixed config of hyperparameters and environment-variables
	into two dictionaries for the two configurations.

	Args:
		config (dict): The potentially mixed configuration.

	Returns:
		dict: The hyperparameter_config
		dict: The environment_config

	Raises:
		AssertionError: If the user provides a key that should not exist.
	"""
	top_level_hyperparameter = HyperparameterConfigValidator.get_required_fields('top-dict')
	top_level_environment = EnvironmentConfig.get_required_fields('top-dict')

	hyperparameter_config = {}
	environment_config = {}

	for key, value in config.items():
		if key in top_level_hyperparameter.keys():
			hyperparameter_config[key] = value
		elif key in top_level_environment.keys():
			environment_config[key] = value
		else:
			raise AssertionError(f'Your config contains an invalid key: {key}')

	validate_sub_keys(HyperparameterConfigValidator, hyperparameter_config, top_level_hyperparameter)
	validate_sub_keys(EnvironmentConfig, environment_config, top_level_environment)

	return hyperparameter_config, environment_config


if __name__ == '__main__':  # pragma: no cover
	test_config_rl = {
		'config_type': 'rl',
		'batch_size': 32,
		'replay_size': 100000,
		'learning_rate': 1e-6,
		'sync_target_frames': 1000,
		'replay_start_size': 10000,
		'epsilon_decay_last_frame': 75000,
		'epsilon_start': 1.0,
		'epsilon_final': 0.1
	}
	test_config_market = {
		'config_type': 'market',
		'max_storage': 100,
		'episode_length': 50,
		'max_price': 10,
		'max_quality': 50,
		'production_price': 3,
		'storage_cost_per_product': 0.1
		}
	test_config_environment = {
		'config_type': 'environment',
		'episodes': 5,
		'agents': [
			{
				'name': 'CE Rebuy Agent (QLearning)',
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent',
				'argument': ''
			},
			{
				'name': 'CE Rebuy Agent (QLearaning)',
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent',
				'argument': ''
			}
		]
	}
	print('Testing config validation...')
	print(validate_config(test_config_rl, False))
	print(validate_config(test_config_market, False))
	print(validate_config(test_config_environment, False))
