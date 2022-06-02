import os

import pytest
import utils_tests as ut_t

import recommerce.configuration.config_validation as config_validation
from recommerce.configuration.path_manager import PathManager

env_config_file = os.path.join(PathManager.user_path, 'configuration_files', 'environment_config_training.json')
market_config_file = os.path.join(PathManager.user_path, 'configuration_files', 'market_config.json')
rl_config_file = os.path.join(PathManager.user_path, 'configuration_files', 'q_learning_config.json')


config_environment = ut_t.load_json(env_config_file)
config_market = ut_t.load_json(market_config_file)
config_rl = ut_t.load_json(rl_config_file)

config_environment['config_type'] = 'environment'
config_market['config_type'] = 'sim_market'
config_rl['config_type'] = 'rl'


def setup_function(function):
	print('***SETUP***')
	global config_environment
	global config_market
	global config_rl

	config_environment['config_type'] = 'environment'
	config_market['config_type'] = 'sim_market'
	config_rl['config_type'] = 'rl'


test_valid_config_validation_complete_testcases = [
	config_environment,
	config_market,
	config_rl
]


@pytest.mark.parametrize('config', test_valid_config_validation_complete_testcases)
def test_valid_config_validation_complete(config):
	config_type = config['config_type']
	success, result = config_validation.validate_config(config, True)
	assert success, result
	assert result == ({config_type: config}, None)


test_valid_config_validation_incomplete_testcases = [
	(config_environment, 'agents'),
	(config_market, 'max_price'),
	(config_rl, 'learning_rate')
]


@pytest.mark.parametrize('config, removed_key', test_valid_config_validation_incomplete_testcases)
def test_valid_config_validation_incomplete(config, removed_key):
	# Hacky, thx pytest!
	tested_config = config.copy()
	tested_config = ut_t.remove_key(removed_key, tested_config)
	config_type = tested_config['config_type']
	success, result = config_validation.validate_config(tested_config, False)
	assert success
	assert result == ({config_type: tested_config}, None)
