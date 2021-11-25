import os
from importlib import reload
from unittest.mock import mock_open, patch

import pytest

from .context import utils_rl


# Helper function that returns a mock config.json file/string with the given values
def create_mock_json(gamma='0.99', batch_size='32', replay_size='100000', learning_rate='1e-6', sync_target_frames='1000', replay_start_size='10000', epsilon_decay_last_frame='75000', epsilon_start='1.0', epsilon_final='0.1'):
	return '{\n\t"gamma" : ' + gamma + ',\n' + \
		'\t"batch_size" : ' + batch_size + ',\n' + \
		'\t"replay_size" : ' + replay_size + ',\n' + \
		'\t"learning_rate" : ' + learning_rate + ',\n' + \
		'\t"sync_target_frames" : ' + sync_target_frames + ',\n' + \
		'\t"replay_start_size" : ' + replay_start_size + ',\n' + \
		'\t"epsilon_decay_last_frame" : ' + epsilon_decay_last_frame + ',\n' + \
		'\t"epsilon_start" : ' + epsilon_start + ',\n' + \
		'\t"epsilon_final" : ' + epsilon_final + '\n' + \
		'}'


# Helper function that builds a mock config.json file/string that is missing a specified line
def create_mock_json_with_missing_line(number, json=create_mock_json()):
	lines = json.split('\n')
	final_lines = lines[0:number + 1]
	final_lines += lines[number + 2:len(lines)]
	final_lines[-2] = final_lines[-2].replace(',', '')
	return '\n'.join(final_lines)


# Helper function to test if the mock_file is setup correctly
def check_mock_file(mock_file, json=create_mock_json()):
	path = os.path.dirname(__file__) + os.sep + '...' + os.sep + 'config_rl.json'
	assert (open(path).read() == json)
	mock_file.assert_called_with(path)
	utils_rl.config = utils_rl.load_config(path)


# mock format taken from: https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth
# Test that checks if the config.json is read correctly
def test_reading_file_values():
	json = create_mock_json()
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		check_mock_file(mock_file)
		# Include utils_rl again to make sure the file is read again
		reload(utils_rl)
		# Test all imported values. Extend this test as new values get added!
		assert utils_rl.GAMMA == 0.99
		assert utils_rl.BATCH_SIZE == 32
		assert utils_rl.REPLAY_SIZE == 100000
		assert utils_rl.LEARNING_RATE == 1e-6
		assert utils_rl.SYNC_TARGET_FRAMES == 1000
		assert utils_rl.REPLAY_START_SIZE == 10000
		assert utils_rl.EPSILON_DECAY_LAST_FRAME == 75000
		assert utils_rl.EPSILON_START == 1.0
		assert utils_rl.EPSILON_FINAL == 0.1

	# Test a second time with other values to ensure, that the values are read correctly
	json2 = create_mock_json(learning_rate='1e-4')
	with patch('builtins.open', mock_open(read_data=json2)) as mock_file:
		check_mock_file(mock_file, json2)
		reload(utils_rl)

		assert utils_rl.LEARNING_RATE == 1e-4


# The following variables are input mock-json strings for the test_invalid_values test
# These tests have invalid values in their input file, the import should throw a specific error message

learning_rate_larger_one = (create_mock_json(learning_rate='1.5'), 'learning_rate should be between 0 and 1 (excluded)')
neg_learning_rate = (create_mock_json(learning_rate='0'), 'learning_rate should be between 0 and 1 (excluded)')

# These tests are missing a line in the config file, the import should throw a specific error message
missing_gamma = (create_mock_json_with_missing_line(0), 'your config_rl is missing gamma')
missing_batch_size = (create_mock_json_with_missing_line(1), 'your config_rl is missing batch_size')
missing_replay_size = (create_mock_json_with_missing_line(2), 'your config_rl is missing replay_size')
missing_learning_rate = (create_mock_json_with_missing_line(3), 'your config_rl is missing learning_rate')
missing_sync_target_frames = (create_mock_json_with_missing_line(4), 'your config_rl is missing sync_target_frames')
missing_replay_start_size = (create_mock_json_with_missing_line(5), 'your config_rl is missing replay_start_size')
missing_epsilon_decay_last_frame = (create_mock_json_with_missing_line(6), 'your config_rl is missing epsilon_decay_last_frame')
missing_epsilon_start = (create_mock_json_with_missing_line(7), 'your config_rl is missing epsilon_start')
missing_epsilon_final = (create_mock_json_with_missing_line(8), 'your config_rl is missing epsilon_final')


# All pairs concerning themselves with invalid config.json values should be added to this array to get tested in test_invalid_values
# johann stole learning_rate, learning_rate_larger_one, neg_learning_rate test
array_testing = [
	missing_gamma, missing_batch_size, missing_replay_size, missing_learning_rate, missing_sync_target_frames, missing_replay_start_size, missing_epsilon_decay_last_frame, missing_epsilon_start, missing_epsilon_final, learning_rate_larger_one, neg_learning_rate
]


# Test that checks that an invalid/broken config.json gets detected correctly
@pytest.mark.parametrize('json_values, expected_error_msg', array_testing)
def test_invalid_values(json_values, expected_error_msg):
	json = json_values
	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
		check_mock_file(mock_file, json)
		with pytest.raises(AssertionError) as assertion_info:
			reload(utils_rl)
		assert expected_error_msg in str(assertion_info.value)
