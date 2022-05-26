import json
import os
# from importlib import reload
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import recommerce.configuration.hyperparameter_config as hyperparameter_config
from recommerce.configuration.path_manager import PathManager

q_learning_config_file = os.path.join(PathManager.user_path, 'configuration_files', 'q_learning_config.json')


# Test that checks if the config.json is read correctly
def test_reading_file_values():
	config = hyperparameter_config.HyperparameterConfigLoader.load('q_learning_config')

	assert config.gamma == 0.99
	assert config.batch_size == 8
	assert config.replay_size == 350
	assert config.learning_rate == 1e-6
	assert config.sync_target_frames == 35
	assert config.replay_start_size == 20
	assert config.epsilon_decay_last_frame == 400
	assert config.epsilon_start == 1.0
	assert config.epsilon_final == 0.1


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message
negative_learning_rate = (ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'learning_rate', 0.0),
	'learning_rate should be positive')
large_gamma = (ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'gamma', 1.1),
	'gamma should be between 0 (included) and 1 (included)')
negative_gamma = ((ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'gamma', -1.0),
	'gamma should be between 0 (included) and 1 (included)'))
negative_batch_size = (ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'batch_size', -5),
	'batch_size should be positive')
negative_replay_size = (ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'replay_size', -5),
	'replay_size should be positive')
negative_sync_target_frames = (ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'sync_target_frames', -5),
	'sync_target_frames should be positive')
negative_replay_start_size = (ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'replay_start_size', -5),
	'replay_start_size should be positive')
negative_epsilon_decay_last_frame = (ut_t.replace_field_in_dict(ut_t.load_json(q_learning_config_file), 'epsilon_decay_last_frame', -5),
	'epsilon_decay_last_frame should be positive')


# These tests are missing a line in the config file, the import should throw a specific error message
missing_gamma = (ut_t.remove_key('gamma', ut_t.load_json(q_learning_config_file)), 'your config is missing gamma')
missing_batch_size = (ut_t.remove_key('batch_size', ut_t.load_json(q_learning_config_file)), 'your config is missing batch_size')
missing_replay_size = (ut_t.remove_key('replay_size', ut_t.load_json(q_learning_config_file)), 'your config is missing replay_size')
missing_learning_rate = (ut_t.remove_key('learning_rate', ut_t.load_json(q_learning_config_file)),
	'your config is missing learning_rate')
missing_sync_target_frames = (ut_t.remove_key('sync_target_frames', ut_t.load_json(q_learning_config_file)),
	'your config is missing sync_target_frames')
missing_replay_start_size = (ut_t.remove_key('replay_start_size', ut_t.load_json(q_learning_config_file)),
	'your config is missing replay_start_size')
missing_epsilon_decay_last_frame = (ut_t.remove_key('epsilon_decay_last_frame', ut_t.load_json(q_learning_config_file)),
	'your config is missing epsilon_decay_last_frame')
missing_epsilon_start = (ut_t.remove_key('epsilon_start', ut_t.load_json(q_learning_config_file)),
	'your config is missing epsilon_start')
missing_epsilon_final = (ut_t.remove_key('epsilon_final', ut_t.load_json(q_learning_config_file)),
	'your config is missing epsilon_final')


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
	mock_json = json.dumps(rl_json)
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		with pytest.raises(AssertionError) as assertion_message:
			hyperparameter_config.HyperparameterConfigLoader.load('hyperparameter_config')
		assert expected_message in str(assertion_message.value)
