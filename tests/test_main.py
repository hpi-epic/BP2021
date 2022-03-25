from unittest.mock import mock_open, patch

import pytest

import recommerce.main as main
from recommerce.configuration.path_manager import PathManager

handle_datapath_valid_testcases = [
	'.',
	'..'
]


@pytest.mark.parametrize('datapath', handle_datapath_valid_testcases)
def test_handle_datapath_valid(datapath):
	with patch('builtins.open', mock_open(read_data=PathManager.user_path)), \
		patch('recommerce.configuration.path_manager.PathManager._update_path_file'):
		main.handle_datapath(datapath)


handle_datapath_invalid_testcases = [
	'1:/invalid',
	''
]


@pytest.mark.parametrize('datapath', handle_datapath_invalid_testcases)
def test_handle_datapath_invalid(datapath):
	with patch('builtins.open', mock_open(read_data=PathManager.user_path)):
		with pytest.raises(AssertionError) as assertion_message:
			main.handle_datapath(datapath)
		assert 'The provided path is not a valid directory' in str(assertion_message.value)


def test_handle_datapath_none():
	# patch will make it so that the saved datapath is a MagicMock == invalid path
	with patch('builtins.open', mock_open(read_data='1:/invalid')):
		with pytest.raises(AssertionError) as assertion_message:
			# Pass None as provided path i.e. no path provided by the user
			main.handle_datapath(None)
		assert 'The current saved data path is invalid:' in str(assertion_message.value)


handle_command_testcases = [
	None,
	'training',
	'exampleprinter',
	'agent_monitoring'
]


@pytest.mark.slow
@pytest.mark.parametrize('command', handle_command_testcases)
def test_handle_command(command):
	with patch('recommerce.rl.training.SummaryWriter'), \
		patch('recommerce.rl.q_learning.q_learning_agent.QLearningAgent.save'):
		main.handle_command(command)
