from unittest.mock import patch

import docker_manager
import pytest

mock_port_mapping = {
	'2d680af1e272e0573d44f0adccccf03361a1c4e8db98c540e03ac84f9d9c4e3c': 6006,
	'c818251ed4088168b51b5677082c1bdf87200fbdb87bab25a7d2faca30ffac6e': 6008,
	'0dcecfa02fcd34588805af5c540ff0e912102a6d91f6f3ee8391af42b8a6831b': 6009
}
with patch('docker_manager.docker'), \
	patch('docker_manager.DockerManager._update_port_mapping'):
	manager = docker_manager.DockerManager()

# Remember to ALWAYS do:
# patch('docker_manager.docker') \
# patch('docker_manager.DockerManager._update_port_mapping'):


def setup_function(function):
	# we mock the mapping initialization since it checks for the currently running containers
	with patch('docker_manager.docker'), \
		patch('docker_manager.DockerManager._update_port_mapping'):
		global manager
		# reset the instance for our tests, as we always want a fresh one
		docker_manager.DockerManager._instance = None
		manager = docker_manager.DockerManager()
		manager._port_mapping = mock_port_mapping


def get_generator():
	"""
	Return a simple stream generator for the tests.
	"""
	yield 1


correct_docker_info_init_testcases = [
	('abc', 'running', 6006, get_generator()),
	('abc', 'running', 6006, None),
	('abc', 'running', '6006', get_generator()),
	('abc', 'running', '6006', None),
	('abc', 'running', True, get_generator()),
	('abc', 'running', False, None),
	('abc', 'running', None, get_generator()),
	('abc', 'running', None, None)
]


@pytest.mark.parametrize('id, status, data, stream', correct_docker_info_init_testcases)
def test_correct_docker_info_initialization(id, status, data, stream):
	docker_manager.DockerInfo(id=id, status=status, data=data, stream=stream)


incorrect_docker_info_init_testcases = [
	(123, 'running', None, None, 'id must be a string'),
	(True, 'running', None, None, 'id must be a string'),
	(None, 'running', None, None, 'id must be a string'),
	('abc', 123, None, None, 'status must be a string'),
	('abc', False, None, None, 'status must be a string'),
	('abc', None, None, None, 'status must be a string'),
	('abc', 'running', get_generator(), None, 'data must be a string, bool or int'),
	('abc', 'running', ['text'], None, 'data must be a string, bool or int'),
	('abc', 'running', ('text', 123), None, 'data must be a string, bool or int'),
	('abc', 'running', None, 123, 'stream must be a stream Generator (GeneratorType)'),
	('abc', 'running', None, 'text', 'stream must be a stream Generator (GeneratorType)')
]


@pytest.mark.parametrize('id, status, data, stream, expected_message', incorrect_docker_info_init_testcases)
def test_incorrect_docker_info_initialization(id, status, data, stream, expected_message):
	with patch('docker_manager.docker'), \
		patch('docker_manager.DockerManager._update_port_mapping'):
		with pytest.raises(AssertionError) as assertion_message:
			docker_manager.DockerInfo(id=id, status=status, data=data, stream=stream)
		assert expected_message in str(assertion_message.value)


def test_docker_manager_is_singleton():
	with patch('docker_manager.docker'), \
		patch('docker_manager.DockerManager._update_port_mapping'):
		manager2 = docker_manager.DockerManager()
		assert manager is manager2


def test_port_mapping_initialization():
	with patch('docker_manager.docker'), \
		patch('docker_manager.DockerManager._update_port_mapping'):
		assert len(manager._port_mapping) == 3
		assert '2d680af1e272e0573d44f0adccccf03361a1c4e8db98c540e03ac84f9d9c4e3c' in manager._port_mapping
		assert manager._port_mapping['c818251ed4088168b51b5677082c1bdf87200fbdb87bab25a7d2faca30ffac6e'] == 6008


def test_allowed_commands_is_up_to_date():
	assert set(manager._allowed_commands) == {'training', 'exampleprinter', 'agent_monitoring'}, \
		'The set of allowed commands has changed, please update this and all the other tests!'


invalid_start_parameter_testcases = [
	({'environment': {'task': 'fasel'}, 'hyperparameter': {'test': 'fasel'}},
		{'id': 'No container was started', 'status': 'Command not allowed: fasel'}),
	({'environment': {'task': 'fasel'}}, {'id': 'No container was started', 'status': 'The config is missing the \"hyperparameter\"-field'}),
	({'hyperparameter': {'task': 'fasel'}}, {'id': 'No container was started', 'status': 'The config is missing the \"environment\"-field'})
]


@pytest.mark.parametrize('test_config, expected_docker_info_params', invalid_start_parameter_testcases)
def test_start_with_invalid_command(test_config, expected_docker_info_params):
	expected_docker_info = docker_manager.DockerInfo(**expected_docker_info_params)

	actual_docker_info = docker_manager.DockerManager().start(test_config, 2)
	assert expected_docker_info == actual_docker_info


def test_start_but_no_image():
	test_config = {'environment': {'task': 'training'}, 'hyperparameter': {'bal': 'fasel'}}
	expected_docker_info = docker_manager.DockerInfo(id='No container was started', status='Image build failed')

	with patch('docker_manager.DockerManager._confirm_image_exists') as image_exists_mock:
		image_exists_mock.return_value = None

		actual_docker_info = docker_manager.DockerManager().start(test_config, 2)
		assert expected_docker_info == actual_docker_info
		image_exists_mock.assert_called_once()


invalid_create_container_status_testcases = [
	({'id': 'test', 'status': 'Image not found bla'}),
	({'id': 'test', 'status': 'test status', 'data': False})
]


@pytest.mark.parametrize('docker_info_mock_parameter', invalid_create_container_status_testcases)
def test_start_create_container_failed(docker_info_mock_parameter):
	test_config = {'environment': {'task': 'training'}, 'hyperparameter': {'bal': 'fasel'}}
	expected_docker_info = docker_manager.DockerInfo(**docker_info_mock_parameter)

	with patch('docker_manager.DockerManager._confirm_image_exists') as image_exists_mock, \
		patch('docker_manager.DockerManager._create_container') as create_container_mock, \
		patch('docker_manager.DockerManager.remove_container') as remove_container_mock:
		image_exists_mock.return_value = '12345'
		create_container_mock.return_value = docker_manager.DockerInfo(**docker_info_mock_parameter)

		actual_docker_info = docker_manager.DockerManager().start(test_config, 2)

		assert expected_docker_info == actual_docker_info
		image_exists_mock.assert_called_once()
		remove_container_mock.assert_called_once_with('test')


def test_start_container_works():
	test_config = {'environment': {'task': 'training'}, 'hyperparameter': {'bal': 'fasel'}}
	docker_info = docker_manager.DockerInfo(id='test1', status='have a wonderful day')

	with patch('docker_manager.DockerManager._confirm_image_exists') as image_exists_mock, \
		patch('docker_manager.DockerManager._create_container') as create_container_mock, \
		patch('docker_manager.DockerManager.remove_container') as remove_container_mock, \
		patch('docker_manager.DockerManager._start_container') as start_container_mock:
		image_exists_mock.return_value = '12345'
		create_container_mock.return_value = docker_info
		actual_docker_infos = docker_manager.DockerManager().start(test_config, 2)

		assert 2 == len(actual_docker_infos)
		image_exists_mock.assert_called_once()
		remove_container_mock.assert_not_called()
		start_container_mock.assert_called()
