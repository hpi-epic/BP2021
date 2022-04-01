from recommerce.configuration.environment_config import EnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfig


def validate_sub_keys(config_class, config: dict, top_level_keys: dict):
	"""
	Utility function that validates if a given config contains only allowed keys.
	Can be used recursively for dictionaries within dictionaries.
	"Unisex": Works for both HyperparameterConfig and EnvironmentConfig.

	Args:
		config_class (HyperparameterConfig or EnvironmentConfig): The config class from which to get the required fields.
		config (dict): The config given by the user.
		top_level_keys (dict): The keys of the current top-level. Their values indicate if there is another dictionary expected for that key.

	Raises:
		AssertionError: If the given config contains a key that is invalid.
	"""
	for key, _ in config.items():
		# TODO: Remove this workaround with the agent-rework in the config files
		if key == 'agents':
			for agent_key in config['agents'].keys():
				assert all(this_key in {'agent_class', 'argument'} for this_key in config['agents'][agent_key]), \
					f'an invalid key for agents was provided: {config["agents"][agent_key].keys()}'
		# the key is key of a dictionary in the config
		elif top_level_keys[key]:
			assert isinstance(config[key], dict), f'The value of this key must be of type dict: {key}, but was {type(config[key])}'
			# these are the valid keys that sub-key can have as keys in the dictionary
			key_fields = config_class.get_required_fields(config_class, key)
			# check that only valid keys were given by the user
			for sub_key, _ in config[key].items():
				if sub_key not in key_fields.keys():
					raise AssertionError(f'The key "{sub_key}" should not exist within a {config_class.__name__} config (was checked at sub-key "{key}")')
			# if there is an additional layer of dictionaries, check it recursively
			validate_sub_keys(config_class, config[key], key_fields)


def split_combined_config(config: dict) -> tuple:
	"""
	Utility function that splits a potentially combined config of hyperparameters and environment-variables
	into two dictionaries for the two configurations.

	Args:
		config (dict): The potentially combined configuration.

	Returns:
		dict: The hyperparameter_config
		dict: The environment_config

	Raises:
		AssertionError: If the user provides a key that should not exist.
	"""
	top_level_hyperparameter = HyperparameterConfig.get_required_fields(HyperparameterConfig, 'top-level')
	top_level_environment = EnvironmentConfig.get_required_fields(EnvironmentConfig, 'top-level')

	hyperparameter_config = {}
	environment_config = {}

	for key, value in config.items():
		if key in top_level_hyperparameter.keys():
			hyperparameter_config[key] = value
		elif key in top_level_environment.keys():
			environment_config[key] = value
		else:
			raise AssertionError(f'This key is unknown: {key}')

	validate_sub_keys(HyperparameterConfig, hyperparameter_config, top_level_hyperparameter)
	validate_sub_keys(EnvironmentConfig, environment_config, top_level_environment)

	return hyperparameter_config, environment_config


def check_config_types(hyperparameter_config: dict, environment_config: dict) -> None:
	"""
	Utility function that checks (incomplete) config dictionaries for their correct types.

	Args:
		hyperparameter_config (dict): The config containing hyperparameter_config-keys.
		environment_config (dict): The config containing environment_config-keys.

	Raises:
		AssertionError: If one of the values has the wring type.
	"""
	# check types for hyperparameter_config
	if 'rl' in hyperparameter_config:
		HyperparameterConfig.check_rl_types(HyperparameterConfig, hyperparameter_config['rl'], False)
	if 'sim_market' in hyperparameter_config:
		HyperparameterConfig.check_sim_market_types(HyperparameterConfig, hyperparameter_config['sim_market'], False)

	# check types for environment_config
	EnvironmentConfig.check_types(EnvironmentConfig, environment_config, False)


if __name__ == '__main__':  # pragma: no cover
	test_config = {
		'rl': {
			'batch_size': 32,
			'replay_size': 100000,
			'learning_rate': 1e-6,
			'sync_target_frames': 1000,
			'replay_start_size': 10000,
			'epsilon_decay_last_frame': 75000,
			'epsilon_start': 1.0,
			'epsilon_final': 0.1
		},
		'sim_market': {
			'max_storage': 100,
			'episode_length': 50,
			'max_price': 10,
			'max_quality': 50,
			'production_price': 3,
			'storage_cost_per_product': 0.1
		},
		'episodes': 5,
		'agents': {
			'CE Rebuy Agent (QLearning)': {
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
				'argument': ''
			},
			'CE Rebuy Agent (QLearaning)': {
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
				'argument': ''
			}
		}
	}
	hyper, env = split_combined_config(test_config)
	check_config_types(hyper, env)
