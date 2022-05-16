import json
import logging
import os
import shutil
import tarfile
import time
from datetime import datetime
from itertools import count, filterfalse
from types import GeneratorType

import docker
from container_db_manager import ContainerDB
from docker.models.containers import Container
from torch.cuda import is_available
from utils import setup_logging

IMAGE_NAME = 'recommerce'


class DockerInfo():
	"""
	This class encapsules the return values for the REST API.
	"""

	def __init__(self, id: str, status: str, data=None, stream: GeneratorType = None) -> None:
		"""
		Args:
			id (str): The sha256 id of the container.
			status (str): Status of the container.
			data ([str, bool, int], optional): Any other data, dependent on the function called this differs.
			stream (stream generator, optional): A stream generator.
		"""
		assert isinstance(id, str), f'id must be a string: {id}'
		assert isinstance(status, str), f'status must be a string: {status}'
		assert isinstance(data, (str, bool, int, type(None))), f'data must be a string, bool or int: {data}'
		assert isinstance(stream, (GeneratorType, type(None))), f'stream must be a stream Generator (GeneratorType): {stream} ({type(stream)})'
		self.id = id
		self.status = status
		self.data = data
		self.stream = stream

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, DockerInfo):
			# don't attempt to compare against unrelated types
			return False
		return self.id == other.id \
			and self.status == other.status \
			and self.data == other.data \
			and self.stream == other.stream


class DockerManager():
	"""
	The DockerManager, implemented as a singleton, is responsible for communicating with Docker.

	It starts containers and performs predefined operations on them, such as starting a training session.
	"""
	_instance = None
	_client = None
	_observers = []
	# This is a list of commands that should be supported by our docker implementation
	_allowed_commands = ['training', 'exampleprinter', 'agent_monitoring']
	# dictionary of container_id:host-port pairs
	_port_mapping = {}
	_container_db = ContainerDB()
	setup_logging('docker_manager')

	def __new__(cls):
		"""
		This function makes sure that the `DockerManager` is a singleton.

		Returns:
			DockerManager: The DockerManager instance.
		"""
		if cls._instance is None:
			logging.info('A new instance of DockerManager is being initialized')
			cls._instance = super(DockerManager, cls).__new__(cls)
			cls._client = cls._get_client()

		if cls._client is not None:
			cls._update_port_mapping()
		return cls._instance

	def check_health_of_all_container(self) -> tuple:
		"""
		Checks health of all containers, and collects exited container and their status code as tuples in a Docker Info

		Returns:
			tuple: first value indecating, if a container has exited, second is a DockerInfo containing exited container
		"""
		exited_recommerce_containers = list(self._get_client().containers.list(filters={'label': 'recommerce', 'status': 'exited'}))
		exited_container = []
		for container in exited_recommerce_containers:
			exited_container += [(container.id, docker.APIClient().inspect_container(container.id)['State']['ExitCode'])]
		self._container_db.they_are_exited(exited_container)
		all_container_ids = ';'.join([str(container_id) for container_id, _ in exited_container])
		all_container_status = ';'.join([str((container_id, exit_code)) for container_id, exit_code in exited_container])
		return exited_recommerce_containers != [], DockerInfo(id=all_container_ids, status=all_container_status)

	def start(self, config: dict, count: int, is_webserver: bool) -> DockerInfo or list:
		"""
		To be called by the REST API. Create and start a new docker container from the image of the specified command.

		Args:
			config (dict): The combined hyperparameter_config and environment_config_command dicts that should be sent to the container.
			count (int): number of containers that should be started
			is_webserver (bool): is the user who did the request the official webserver?

		Returns:
			DockerInfo or list: A JSON serializable object containing the error messages if the prerequisite were not met, or a list of
				DockerInfos for the container(s)
		"""
		if 'hyperparameter' not in config:
			return DockerInfo(id='No container was started', status='The config is missing the "hyperparameter"-field')
		if 'environment' not in config:
			return DockerInfo(id='No container was started', status='The config is missing the "environment"-field')

		command_id = config['environment']['task']

		if command_id not in self._allowed_commands:
			logging.warning(f'Command with ID {command_id} not allowed')
			return DockerInfo(id='No container was started', status=f'Command not allowed: {command_id}')

		if not self._confirm_image_exists():
			return DockerInfo(id='No container was started', status='Image build failed')

		current_time = datetime.now()
		all_container_infos = []
		for _ in range(count):
			# start a container for the image of the requested command
			container_info: DockerInfo = self._create_container(command_id, config, use_gpu=is_available())
			if 'Image not found' in container_info.status or container_info.data is False:
				# something is wrong with our container
				self.remove_container(container_info.id)
				return container_info
			# the container is fine, we can start the container now
			all_container_infos += [self._start_container(container_info.id)]
		self._container_db.insert(all_container_infos, current_time, is_webserver, config)
		return all_container_infos

	def health(self, container_id: str) -> DockerInfo:
		"""
		To be called by the REST API. Return the status of the specified container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		logging.info(f'Checking health status for: {container_id}')
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		self._container_db.has_been_health_checked(container_id)
		if container.status == 'exited':
			return DockerInfo(container_id, status=f'exited ({docker.APIClient().inspect_container(container.id)["State"]["ExitCode"]})')

		return DockerInfo(container_id, status=container.status)

	def pause(self, container_id: str) -> DockerInfo:
		"""
		To be called by the REST API. Pauses the container if it is not already paused.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		if container.status == 'exited':
			return DockerInfo(container_id, status=f'exited ({container.wait()["StatusCode"]})')
		elif container.status == 'paused':
			return DockerInfo(container_id, status='paused')

		try:
			container.pause()
			# Reload the attributes to get the correct status
			container.reload()
			self._container_db.has_been_paused(container_id)
			return DockerInfo(id=container_id, status=container.status)
		except docker.errors.APIError as error:
			return DockerInfo(container_id, status=f'APIError encountered while pausing container.\n{error}')

	def unpause(self, container_id: str) -> DockerInfo:
		"""
		To be called by the REST API. Unpauses the container if it is paused.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		if container.status == 'exited':
			return DockerInfo(container_id, status=f'exited ({container.wait()["StatusCode"]})')
		elif container.status != 'paused':
			return DockerInfo(container_id, status=container.status)

		try:
			container.unpause()
			# Reload the attributes to get the correct status
			container.reload()
			self._container_db.has_been_unpaused(container_id)
			return DockerInfo(id=container_id, status=container.status)
		except docker.errors.APIError as error:
			return DockerInfo(container_id, status=f'APIError encountered while unpausing container.\n{error}')

	def start_tensorboard(self, container_id: str) -> DockerInfo:
		"""
		To be called by the REST API. Start a tensorboard session on the specified container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A DockerInfo object containing the id of the container, the status and a link to the tensorboard in the data field.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		if container.status != 'running':
			return DockerInfo(container_id, status='Container is not running. Download the data and start a tensorboard locally.')

		logging.info(f'Starting tensorboard for: {container_id}')
		container.exec_run(cmd='tensorboard serve --host 0.0.0.0 --logdir ./results/runs', detach=True)
		port = self._port_mapping[container.id]
		self._container_db.has_got_tensorboard(container_id)
		return DockerInfo(container_id, status=container.status, data=str(port))

	def get_container_logs(self, container_id: str, timestamps: bool, stream: bool, tail: int) -> DockerInfo:
		"""
		To be called by the REST API. Return the current logs of the container as a string.

		Args:
			container_id (str): The id of the container.
			timestamps (bool): Whether or not timestamps should be included in the logs.
			stream (bool): Whether to stream the logs instead of directly retrieving them.
			tail (int): How many lines at the end of the logs should be returned. int or 'all'.

		Returns:
			DockerInfo: A DockerInfo object containing the id of the container, the status and the logs of the container in the data field,
				or if stream is True, the stream generator in the stream field.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		logging.info(f'Getting logs for {container_id}...')

		logs = container.logs(stream=stream, timestamps=timestamps, tail=tail,
			stderr=docker.APIClient().inspect_container(container.id)['State']['ExitCode'] != 0)
		self._container_db.has_got_logs(container_id)
		if stream:
			return DockerInfo(container_id, status=container.status, stream=logs)
		else:
			return DockerInfo(container_id, status=container.status, data=logs.decode('utf-8'))

	def get_container_data(self, container_id: str, container_path: str) -> DockerInfo:
		"""
		To be called by the REST API.
		Return a data stream object that matches a .tar archive as well as a filename derived from the current time and the provided file-path.

		Args:
			container_id (str): The id of the container.
			container_path (str): The path in the container from which to get the data. Defaults to '/app/results'.

		Returns:
			DockerInfo: Contains the container_id, a filename in the data field and a stream generator matching the .tar archive.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')
		try:
			bits, _ = container.get_archive(path=container_path)
			self._container_db.has_got_data(container_id)
			return DockerInfo(container_id, status=container.status,
				data=f'archive_{container_path.rpartition("/")[2]}_{time.strftime("%b%d_%H-%M-%S")}', stream=bits)
		except docker.errors.NotFound:
			return DockerInfo(container_id, status=f'The requested path does not exist on the container: {container_path}')

	def get_statistic_data(self, wants_system_data: bool):
		return DockerInfo(id='not given', status='success', data=self._container_db.get_csv_data(wants_system_data))

	def remove_container(self, container_id: str) -> DockerInfo:
		"""
		To be called by the REST API. Stop and remove a container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the container.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		container_info = self._stop_container(container_id)
		if container_info.status != 'exited':
			logging.warning(f'Container not stopped successfully. Status: {container_info.status}')
			return DockerInfo(id=container_id, status=f'Container not stopped successfully. Status: {container_info.status}')

		logging.info(f'Removing container: {container_id}')
		try:
			exit_code = self._get_container_exit_code(container)
			container.remove()
			# update the local port mapping
			self._update_port_mapping()

			return DockerInfo(id=container_id, status=f'removed ({exit_code})')
		except docker.errors.APIError as error:
			return DockerInfo(id=container_id, status=f'APIError encountered while removing container.\n{error}')

	def ping(self) -> bool:
		"""
		Wrapper around docker.ping() that pings the docker server to see if it is running.

		Returns:
			bool: If the server is running or not.
		"""
		logging.info('Pinging docker server...')
		try:
			return self._get_client().ping()
		except Exception:
			logging.warning('Docker server is not responding!')
			return False

	# PRIVATE METHODS
	@classmethod
	def _get_client(cls) -> docker.DockerClient:
		"""
		"Wrapper" around `cls._client`. If the `cls._client` is already set, return it.
		Otherwise, try and get a new client from docker and set it as the `cls._client`.
		If docker is unavailable, `cls.client` will stay as `None`.

		This function makes it possible for docker to become unavailable and available again without crashing the DockerManager.

		Returns:
			docker.DockerClient: The docker client, or None if docker is unavailable.
		"""
		if cls._client is not None:
			return cls._client
		try:
			cls._client = docker.from_env()
		except docker.errors.DockerException:
			cls._client = None
		return cls._client

	def _confirm_image_exists(self, update: bool = False) -> str:
		"""
		Find out if the IMAGE_NAME image exists. If not, the image will be built.

		Args:
			update (bool): Whether or not to always build/update an image for this id. Defaults to False.

		Returns:
			str: The id of the image or None if the build failed.
		"""

		# Get the image tag of all images on the system.
		all_images = self._get_client().images.list()
		tagged_images = [image.tags[0].rsplit(':')[0] for image in all_images if len(image.tags)]

		if len(all_images) != len(tagged_images):
			logging.info('You have untagged images and may want to remove them:')
			for image in all_images:
				if len(image.tags) == 0:
					logging.info(image.id)

		if update:
			logging.info(f'{IMAGE_NAME} image will be created/updated.')
			return self._build_image()

		if IMAGE_NAME not in tagged_images:
			logging.info(f'{IMAGE_NAME} image does not exist and will be created')
			return self._build_image()
		logging.info(f'{IMAGE_NAME} image already exists')
		return self._get_client().images.get(IMAGE_NAME).id[7:]

	def _build_image(self) -> str:
		"""
		Build an image for the recommerce application, and name it IMAGE_NAME.

		Returns:
			str: The id of the image or None if the build failed.
		"""
		# https://docker-py.readthedocs.io/en/stable/images.html
		logging.info(f'Building {IMAGE_NAME} image')

		# Find out if an image with the name already exists to remove it afterwards
		try:
			old_img = self._get_client().images.get(IMAGE_NAME)
		except docker.errors.ImageNotFound:
			old_img = None
		try:
			# Using the low-level API to be able to get live-logs
			logs = docker.APIClient().build(path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
				tag=IMAGE_NAME, forcerm=True, network_mode='host', decode=True)
			for output in logs:
				if 'stream' in output:
					output_str = output['stream'].strip('\r\n').strip('\n')
					logging.info(output_str)
			img = self._get_client().images.get(IMAGE_NAME)
		except docker.errors.BuildError or docker.errors.APIError as error:
			logging.error(f'An error occurred while building the {IMAGE_NAME} image\n{error}')
			return None
		if old_img is not None and old_img.id != img.id:
			logging.warning(f'\nA {IMAGE_NAME} image already exists, it will be overwritten')
			self._get_client().images.remove(old_img.id[7:])
		# return id without the 'sha256:'-prefix
		return img.id[7:]

	def _create_container(self, command_id: str, config: dict, use_gpu: bool = True) -> DockerInfo:
		"""
		Create a container for the given image.

		Args:
			command_id (str): The command to run in the container.
			config (dict): A dict containing parameters for the simulation.
			use_gpu (bool) Whether or not to request access to GPU's. Defaults to True.

		Returns:
			DockerInfo: A DockerInfo object with id and status set.
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		logging.info(f'Creating container for command: {command_id}')

		# first update the port mapping in case containers were added/removed without our knowledge
		self._update_port_mapping()
		# find the next available port to map to 6006 in the container
		used_port = next(filterfalse(set(self._port_mapping.values()).__contains__, count(6006)))
		# create a device request to use all available GPU devices with compute capabilities
		if use_gpu:
			device_request_gpu = docker.types.DeviceRequest(driver='nvidia', count=-1, capabilities=[['compute']])
			try:
				container: Container = self._get_client().containers.create(IMAGE_NAME,
					detach=True,
					labels=[IMAGE_NAME],
					ports={'6006/tcp': used_port},
					entrypoint=f'recommerce -c {command_id}',
					device_requests=[device_request_gpu])
			except docker.errors.ImageNotFound as error:
				return DockerInfo(id=command_id, status=f'Image not found.\n{error}')
		else:
			try:
				container: Container = self._get_client().containers.create(IMAGE_NAME,
					detach=True,
					labels=[IMAGE_NAME],
					ports={'6006/tcp': used_port},
					entrypoint=f'recommerce -c {command_id}')
			except docker.errors.ImageNotFound as error:
				return DockerInfo(id=command_id, status=f'Image not found.\n{error}')

		# add the new container to the port mapping
		self._port_mapping.update({container.id: used_port})

		upload_info = self._upload_config(container.id, command_id, config)
		if not upload_info.data:
			logging.warning('Failed to upload configuration file!')
		return upload_info

	def _start_container(self, container_id: str) -> DockerInfo:
		"""
		Start a container for the given image.

		Args:
			container_id (str): The id of the image to start the container for.

		Returns:
			DockerInfo: A DockerInfo object with the id and the status of the started docker container.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(id=container_id, status='Container not found.')

		if container.status == 'running':
			logging.info(f'Container is already running: {container_id}')
			return DockerInfo(id=container_id, status='running')

		logging.info(f'Starting container: {container_id}')
		try:
			container.start()
			# Reload the attributes to get the correct status
			container.reload()
			return DockerInfo(id=container_id, status=container.status, data=self._port_mapping[container.id])
		except docker.errors.APIError as error:
			container.remove()
			return DockerInfo(id=container_id, status=f'APIError encountered while starting container.\n{error}')

	def _get_container(self, container_id: str) -> Container:
		"""
		Get the container for the given id. If the container does not exist, return None.

		Args:
			container_id (str): The id of the container.

		Returns:
			docker.models.containers.Container: The container for the given id or None if the container does not exist.
		"""
		try:
			return self._get_client().containers.get(container_id)
		except docker.errors.NotFound:
			return None

	def _get_container_exit_code(self, container: Container) -> str:
		try:
			exit_code = container.wait()['StatusCode']
		except docker.errors.APIError as error:
			exit_code = f'could not get, {error}'
		return exit_code

	def _stop_container(self, container_id: str) -> DockerInfo:
		"""
		Stop a running container.

		After 10 seconds of no response, the container will be killed.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the container.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found.')

		logging.info(f'Stopping container: {container_id}')
		before_stop = container.status
		try:
			container.stop(timeout=10)
			# Reload the attributes to get the correct status
			container.reload()
			self._container_db.has_been_stopped(container_id, before_stop, container.status, self._get_container_exit_code(container))
			return DockerInfo(id=container_id, status=container.status)
		except docker.errors.APIError as error:
			return DockerInfo(container_id, status=f'APIError encountered while stopping container.\n{error}')

	def _upload_config(self, container_id: str, command_id: str, config_dict: dict) -> DockerInfo:
		"""
		Upload the config-files to the specified container.

		Args:
			container_id (str): The id of the container.
			command_id (str): The command to run in the container.
			config_dict (dict): The config dictionary to upload.

		Returns:
			DockerInfo: A DockerInfo object with the id of the container and the status of the upload in the data field.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(id=container_id, status='Container not found.')

		logging.info('Copying config files into container...')
		# create a directory to store the files safely
		os.makedirs('config_tmp', exist_ok=True)
		os.chdir('config_tmp')

		# this folder will contain all config files
		os.makedirs('configuration_files', exist_ok=True)

		# write dict to json
		with open(os.path.join('configuration_files', 'hyperparameter_config.json'), 'w') as config_json:
			config_json.write(json.dumps(config_dict['hyperparameter']))
		with open(os.path.join('configuration_files', f'environment_config_{command_id}.json'), 'w') as config_json:
			config_json.write(json.dumps(config_dict['environment']))

		# put config files in tar archive
		with tarfile.open('configuration_files.tar', 'w') as tar:
			try:
				tar.add('configuration_files')
			finally:
				tar.close()

		# uploading the tar to the container
		upload_ok = False
		with open('configuration_files.tar', 'rb') as tar_archive:
			upload_ok = container.put_archive(path='/app', data=tar_archive)

		# remove obsolete files
		if upload_ok:
			os.chdir('..')
			shutil.rmtree('config_tmp')
		logging.info('Copying config files complete')
		return DockerInfo(id=container_id, status=container.status, data=upload_ok)

	@classmethod
	def _update_port_mapping(cls):
		"""
		Update the cls._port_mapping dictionary.

		Gets all containers tagged with IMAGE_NAME and find the port forwarded from 6006.
		"""
		# Get all RUNNING containers with the IMAGE_NAME label
		# we don't care about already exited containers, since we can't see the tensorboard anyways
		running_recommerce_containers = list(cls._get_client().containers.list(filters={'label': IMAGE_NAME}))
		# Get the port mapped to '6006/tcp' within the container
		occupied_ports = [int(container.ports['6006/tcp'][0]['HostPort']) for container in running_recommerce_containers]
		# Create a dictionary of container_id: mapped port
		cls._port_mapping = dict(zip([container.id for container in running_recommerce_containers], occupied_ports))


if __name__ == '__main__':  # pragma: no cover
	manager = DockerManager()
	print(manager._confirm_image_exists(update=True), '\n')
