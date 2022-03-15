from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import alpha_business.configuration.hyperparameter_config as hyperparameter_config


def teardown_module(module):
	print('***TEARDOWN***')
	reload(hyperparameter_config)


def import_config() -> hyperparameter_config.HyperparameterConfig:
	"""
	Reload the hyperparameter_config file to update the config variable with the mocked values.

	Returns:
		HyperparameterConfig: The config object.
	"""
	reload(hyperparameter_config)
	return hyperparameter_config.config


# mock format taken from:
# https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth
# Test that checks if the config.json is read correctly
def test_reading_file_values():
	json = ut_t.create_hyperparameter_mock_json()
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)

		config = import_config()

		assert config.gamma == 0.99
		assert config.batch_size == 32
		assert config.replay_size == 100000
		assert config.learning_rate == 1e-6
		assert config.sync_target_frames == 1000
		assert config.replay_start_size == 10000
		assert config.epsilon_decay_last_frame == 75000
		assert config.epsilon_start == 1.0
		assert config.epsilon_final == 0.1

	# Test a second time with other values to ensure that the values are read correctly
	json2 = ut_t.create_hyperparameter_mock_json(rl=ut_t.create_hyperparameter_mock_json_rl(learning_rate='1e-4'))
	with patch('builtins.open', mock_open(read_data=json2)) as mock_file:
		ut_t.check_mock_file(mock_file, json2)

		config = import_config()

		assert config.learning_rate == 1e-4


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message
learning_rate_larger_one = (ut_t.create_hyperparameter_mock_json_rl(learning_rate='1.5'),
	'learning_rate should be between 0 and 1 (excluded)')
negative_learning_rate = (ut_t.create_hyperparameter_mock_json_rl(learning_rate='0'), 'learning_rate should be between 0 and 1 (excluded)')
large_gamma = (ut_t.create_hyperparameter_mock_json_rl(gamma='1'), 'gamma should be between 0 (included) and 1 (excluded)')
negative_gamma = ((ut_t.create_hyperparameter_mock_json_rl(gamma='-1'), 'gamma should be between 0 (included) and 1 (excluded)'))
negative_batch_size = (ut_t.create_hyperparameter_mock_json_rl(batch_size='-5'), 'batch_size should be greater than 0')
negative_replay_size = (ut_t.create_hyperparameter_mock_json_rl(replay_size='-5'),
	'replay_size should be greater than 0')
negative_sync_target_frames = (ut_t.create_hyperparameter_mock_json_rl(sync_target_frames='-5'),
	'sync_target_frames should be greater than 0')
negative_replay_start_size = (ut_t.create_hyperparameter_mock_json_rl(replay_start_size='-5'), 'replay_start_size should be greater than 0')
negative_epsilon_decay_last_frame = (ut_t.create_hyperparameter_mock_json_rl(epsilon_decay_last_frame='-5'),
	'epsilon_decay_last_frame should not be negative')


# These tests are missing a line in the config file, the import should throw a specific error message
missing_gamma = (ut_t.remove_line(0, ut_t.create_hyperparameter_mock_json_rl()), 'your config_rl is missing gamma')
missing_batch_size = (ut_t.remove_line(1, ut_t.create_hyperparameter_mock_json_rl()), 'your config_rl is missing batch_size')
missing_replay_size = (ut_t.remove_line(2, ut_t.create_hyperparameter_mock_json_rl()), 'your config_rl is missing replay_size')
missing_learning_rate = (ut_t.remove_line(3, ut_t.create_hyperparameter_mock_json_rl()),
	'your config_rl is missing learning_rate')
missing_sync_target_frames = (ut_t.remove_line(4, ut_t.create_hyperparameter_mock_json_rl()),
	'your config_rl is missing sync_target_frames')
missing_replay_start_size = (ut_t.remove_line(5, ut_t.create_hyperparameter_mock_json_rl()),
	'your config_rl is missing replay_start_size')
missing_epsilon_decay_last_frame = (ut_t.remove_line(6, ut_t.create_hyperparameter_mock_json_rl()),
	'your config_rl is missing epsilon_decay_last_frame')
missing_epsilon_start = (ut_t.remove_line(7, ut_t.create_hyperparameter_mock_json_rl()), 'your config_rl is missing epsilon_start')
missing_epsilon_final = (ut_t.remove_line(8, ut_t.create_hyperparameter_mock_json_rl()), 'your config_rl is missing epsilon_final')


invalid_values_testcases = [
	missing_gamma,
	missing_batch_size,
	missing_replay_size,
	missing_learning_rate,
	missing_sync_target_frames,
	missing_replay_start_size,
	missing_epsilon_decay_last_frame,
	missing_epsilon_start,
	missing_epsilon_final,
	learning_rate_larger_one,
	negative_learning_rate,
	large_gamma,
	negative_gamma,
	negative_batch_size,
	negative_replay_size,
	negative_sync_target_frames,
	negative_replay_start_size,
	negative_epsilon_decay_last_frame
]


# Test that checks that an invalid/broken config.json gets detected correctly
@pytest.mark.parametrize('rl_json, expected_message', invalid_values_testcases)
def test_invalid_values(rl_json, expected_message):
	json = ut_t.create_hyperparameter_mock_json(rl=rl_json)
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		ut_t.check_mock_file(mock_file, json)
		with pytest.raises(AssertionError) as assertion_message:
			import_config()
		assert expected_message in str(assertion_message.value)
