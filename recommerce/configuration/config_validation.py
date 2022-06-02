# This file contains logic used by the webserver to validate configuration files

from recommerce.configuration.environment_config import EnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigValidator
from recommerce.configuration.utils import get_class


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
		# first check if we got a complete config from the webserver
		# in which case we will have two keys on the top-level
		# this can either be an uploaded complete config, or a config sent when pressing the launch/check button
		if 'environment' in config and 'hyperparameter' in config:
			assert len(config) == 2, 'Your config should not contain keys other than "environment" and "hyperparameter"'
			hyperparameter_config = config['hyperparameter']
			environment_config = config['environment']

			market_class = get_class(environment_config['marketplace'])
			agent_class = get_class(environment_config['agents'][0]['agent_class'])

			HyperparameterConfigValidator.validate_config(hyperparameter_config['sim_market'], market_class)
			HyperparameterConfigValidator.validate_config(hyperparameter_config['rl'], agent_class)
			EnvironmentConfig.check_types(environment_config, environment_config['task'], False, True)

			return True, (hyperparameter_config, environment_config)
		# if the two keys are not present, the config MUST be one of environment, rl, or market
		# this is only the case when uploading a config
		else:
			config_type = find_config_type(config)

			# we can only validate types for the environment_config, as we do not know the agent/market class for the rl/market configs
			if config_type == 'environment':
				# validate that all given values have the correct types
				task = config['task'] if config_is_final else 'None'
				EnvironmentConfig.check_types(config, task, False, config_is_final)

			# the webserver needs another format for the config
			config.pop('config_type')
			if config_type == 'rl':
				return True, ({'rl': config}, None)
			elif config_type == 'sim_market':
				return True, ({'sim_market': config}, None)
			else:
				return True, ({'environment': config}, None)

	except Exception as error:
		return False, str(error)


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
		if config['config_type'] in ['rl', 'sim_market', 'environment']:
			return config['config_type']
		else:
			raise AssertionError(f'the "config_type" key must be one of "rl", "sim_market", "environment" but was {config["config_type"]}')
	except KeyError as e:
		raise AssertionError(f"your config is missing the 'config_type' key, must be one of 'rl', 'sim_market', 'environment': {config}") from e


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
		'config_type': 'sim_market',
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
