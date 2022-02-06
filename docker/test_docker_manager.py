from unittest.mock import patch

import docker_manager

mock_port_mapping = {
	'2d680af1e272e0573d44f0adccccf03361a1c4e8db98c540e03ac84f9d9c4e3c': 6006,
	'c818251ed4088168b51b5677082c1bdf87200fbdb87bab25a7d2faca30ffac6e': 6008,
	'0dcecfa02fcd34588805af5c540ff0e912102a6d91f6f3ee8391af42b8a6831b': 6009
}
manager = docker_manager.DockerManager()

# Remember to ALWAYS patch('docker_manager.docker')


def setup_function(function):
	# we mock the mapping initialization since it checks for the currently running containers
	with patch('docker_manager.docker'), \
		patch('docker_manager.DockerManager._initialize_port_mapping'):
		global manager
		# reset the instance for our tests, as we always want a fresh one
		docker_manager.DockerManager._instance = None
		manager = docker_manager.DockerManager()
		manager._port_mapping = mock_port_mapping


def test_docker_manager_is_singleton():
	with patch('docker_manager.docker'):
		manager2 = docker_manager.DockerManager()
		assert manager is manager2


def test_port_mapping_initialization():
	with patch('docker_manager.docker'):
		assert len(manager._port_mapping) == 3
		assert '2d680af1e272e0573d44f0adccccf03361a1c4e8db98c540e03ac84f9d9c4e3c' in manager._port_mapping
		assert manager._port_mapping['c818251ed4088168b51b5677082c1bdf87200fbdb87bab25a7d2faca30ffac6e'] == 6008
