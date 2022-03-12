import json
import os
import tarfile
import time
from itertools import count, filterfalse
from types import GeneratorType

import docker
from docker.models.containers import Container


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


class DockerManager():
	"""
	The DockerManager, implemented as a singleton, is responsible for communicating with Docker.

	It starts containers and performs predefined operations on them, such as starting a training session.
	"""
	_instance = None
	_client = None
	_observers = []
	# This is a list of commands that should be supported by our docker implementation
	# For each command, there must be a dockerfile of the format 'dockerfile_command' in the docker/dockerfiles folder
	_allowed_commands = ['training', 'exampleprinter', 'monitoring']
	# dictionary of container_id:host-port pairs
	_port_mapping = {}

	def __new__(cls):
		"""
		This function makes sure that the `DockerManager` is a singleton.

		Returns:
			DockerManager: The DockerManager instance.
		"""
		if cls._instance is None:
			print('A new instance of DockerManager is being initialized')
			cls._instance = super(DockerManager, cls).__new__(cls)
			cls._client = docker.from_env()

		cls._initialize_port_mapping(cls)
		return cls._instance

	def start(self, command_id: str, hyperparameter_config: dict) -> DockerInfo:
		"""
		To be called by the REST API. Create and start a new docker container from the image of the specified command.

		Args:
			command_id (str): The command for which to start a container.
			hyperparameter_config (dict): The hyperparameter_config.json to replace the default one with.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		if command_id not in self._allowed_commands:
			print(f'Command with ID {command_id} not allowed')
			return DockerInfo(id=None, status=f'Command not allowed: {command_id}')

		self._confirm_image_exists(command_id)

		# start a container for the image of the requested command
		container_info: DockerInfo = self._create_container(command_id, hyperparameter_config, use_gpu=False)
		if container_info.status.__contains__('Image not found') or container_info.data is False:
			self.remove_container(container_info.id)
			return container_info

		return self._start_container(container_info.id)

	def health(self, container_id: str) -> DockerInfo:
		"""
		To be called by the REST API. Return the status of the specified container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		print(f'Checking health status for: {container_id}')
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		if container.status == 'exited':
			return DockerInfo(container_id, status=f'exited ({container.wait()["StatusCode"]})')

		return DockerInfo(container_id, status=container.status)

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

		print(f'Starting tensorboard for: {container_id}')
		container.exec_run(cmd='tensorboard serve --host 0.0.0.0 --logdir ./results/runs', detach=True)
		port = self._port_mapping[container.id]
		return DockerInfo(container_id, status=container.status, data=f'http://localhost:{port}')

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

		logs = container.logs(stream=stream, timestamps=timestamps, tail=tail)
		if stream:
			return DockerInfo(container_id, status=container.status, stream=logs)
		else:
			return DockerInfo(container_id, status=container.status, data=logs.decode('utf-8'))

	def get_container_data(self, container_id: str, container_path: str) -> DockerInfo:
		"""
		To be called by the REST API.
		Return a data stream object that matches a .tar archive as well as a filename derived from the curent time and the provided file-path.

		Args:
			container_id (str): The id of the container.
			container_path (str): The path in the container from which to get the data. Defaults to '/app/results'.

		Returns:
			DockerInfo: Contains the container_id, a filename in the data field and a stream generator matching the .tar archive.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status='Container not found')

		bits, _ = container.get_archive(path=container_path)
		return DockerInfo(container_id, status=container.status,
			data=f'archive_{container_path.rpartition("/")[2]}_{time.strftime("%b%d_%H-%M-%S")}', stream=bits)

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
			print(f'Container not stopped successfully. Status: {container_info.status}')
			return DockerInfo(id=container_id, status=f'Container not stopped successfully. Status: {container_info.status}')

		print(f'Removing container: {container_id}')
		try:
			exit_code = container.wait()['StatusCode']
			container.remove()
			# update the local port mapping
			self._port_mapping.pop(container.id)
			# remove the port mapping of the old container from the occupied_ports.txt
			with open(os.path.join(os.path.dirname(__file__), 'occupied_ports.txt'), 'w') as port_file:
				pass
			with open(os.path.join(os.path.dirname(__file__), 'occupied_ports.txt'), 'a') as port_file:
				for id, port in self._port_mapping.items():
					port_file.write(f'{id}\n{port}\n')

			return DockerInfo(id=container_id, status=f'removed ({exit_code})')
		except docker.errors.APIError as error:
			return DockerInfo(id=container_id, status=f'APIError encountered while removing container.\n{error}')

	# PRIVATE METHODS
	def _confirm_image_exists(self, command_id: str, update: bool = False) -> str:
		"""
		Find out if the specified image exists.	If not, the image will be built. If the command is invalid, None is returned.

		Args:
			command_id (str): The command for which an image should exist.
			update (bool): Whether or not to always build/update an image for this id. Defaults to False.

		Returns:
			str: The id of the image.
		"""
		if command_id not in self._allowed_commands:
			print(f'Invalid command: {command_id}. No image will be built.')
			return

		# Get the image tag of all images on the system.
		all_images = self._client.images.list()
		tagged_images = [image.tags[0].rsplit(':')[0] for image in all_images if len(image.tags)]

		if len(all_images) != len(tagged_images):
			print('You have untagged images and may want to remove them:')
			for image in all_images:
				if len(image.tags) == 0:
					print(image.id)

		if update:
			print(f'Image for command {command_id} will be created/updated.')
			return self._build_image(command_id=command_id)

		if command_id not in tagged_images:
			print(f'Image does not exist and will be created: {command_id}')
			return self._build_image(command_id=command_id)
		print(f'Image already exists: {command_id}')
		return self._client.images.get(command_id).id[7:]

	def _build_image(self, command_id: str) -> str:
		"""
		Build an image from the dockerfile of the command and name it accordingly.

		Args:
			command_id (str): The command for which to build the image for.

		Returns:
			str: The id of the image.
		"""
		# https://docker-py.readthedocs.io/en/stable/images.html
		# build image from dockerfile and name it accordingly
		print(f'Building image for command: {command_id}')

		# Copy over the correct dockerfile for the command
		with open(os.path.join(os.path.dirname(__file__), os.pardir, 'dockerfile'), 'r') as dockerfile_template:
			template = dockerfile_template.read()
		with open(os.path.join(os.path.dirname(__file__), 'dockerfiles', f'dockerfile_{command_id}'), 'r') as new_dockerfile:
			docker_file = new_dockerfile.read()
		with open(os.path.join(os.path.dirname(__file__), os.pardir, 'dockerfile'), 'w') as dockerfile:
			dockerfile.write(docker_file)

		# Find out if an image with the name already exists to remove it afterwards
		try:
			old_img = self._client.images.get(command_id)
		except docker.errors.ImageNotFound:
			old_img = None
		try:
			img, _ = self._client.images.build(path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
				tag=command_id, forcerm=True, network_mode='host')
		except docker.errors.BuildError or docker.errors.APIError as error:
			print(f'An error occurred while building the image for command: {command_id}\n{error}')
			exit(1)
		# No matter what happens during the build, we reset our default dockerfile
		finally:
			with open(os.path.join(os.path.dirname(__file__), os.pardir, 'dockerfile'), 'w') as dockerfile:
				dockerfile.write(template)

		if old_img is not None and old_img.id != img.id:
			print(f'An image with this name already exists, it will be overwritten: {command_id}')
			self._client.images.remove(old_img.id[7:])
		# return id without the 'sha256:'-prefix
		return img.id[7:]

	def _create_container(self, image_id: str, hyperparameter_config: dict, use_gpu: bool = True) -> DockerInfo:
		"""
		Create a container for the given image.

		Args:
			image_id (str): The id of the image to create the container for.
			hyperparameter_config (str): A json containing parameters for the simulation.
			use_gpu (bool) Whether or not to request access to GPU's. Defaults to True.

		Returns:
			DockerInfo: A DockerInfo object with id and status set.
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		print(f'Creating container for image: {image_id}')

		# find the next available port to map to 6006 in the container
		used_port = next(filterfalse(set(self._port_mapping.values()).__contains__, count(6006)))
		# create a device request to use all available GPU devices with compute capabilities
		if use_gpu:
			device_request_gpu = docker.types.DeviceRequest(driver='nvidia', count=-1, capabilities=[['compute']])
			try:
				container: Container = self._client.containers.create(image_id,
					detach=True,
					ports={'6006/tcp': used_port},
					device_requests=[device_request_gpu])
			except docker.errors.ImageNotFound as error:
				return DockerInfo(id=image_id, status=f'Image not found.\n{error}')
		else:
			try:
				container: Container = self._client.containers.create(image_id, detach=True, ports={'6006/tcp': used_port})
			except docker.errors.ImageNotFound as error:
				return DockerInfo(id=image_id, status=f'Image not found.\n{error}')

		# Add the container.id and used_port to the occupied_ports.txt and the self._port_mapping
		with open(os.path.join(os.path.dirname(__file__), 'occupied_ports.txt'), 'a') as port_file:
			port_file.write(f'{container.id}\n{used_port}\n')
		self._port_mapping.update({container.id: used_port})

		upload_info = self._upload_config(container.id, hyperparameter_config)
		if not upload_info.data:
			print('Failed to upload configuration file!')
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
			print(f'Container is already running: {container_id}')
			return DockerInfo(id=container_id, status='running')

		print(f'Starting container: {container_id}')
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
			return self._client.containers.get(container_id)
		except docker.errors.NotFound:
			return None

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

		print(f'Stopping container: {container_id}')
		try:
			container.stop(timeout=10)
			# Reload the attributes to get the correct status
			container.reload()
			return DockerInfo(id=container_id, status=container.status)
		except docker.errors.APIError as error:
			return DockerInfo(container_id, status=f'APIError encountered while stopping container.\n{error}')

	def _upload_config(self, container_id: str, hyperparameter_config_dict: dict) -> DockerInfo:
		"""
		Upload a file to the specified container.

		Args:
			container_id (str): The id of the container.
			hyperparameter_config_dict (dict): The config dictionary to upload.

		Returns:
			DockerInfo: A DockerInfo object with the id of the container and the status of the upload in the data field.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(id=container_id, status='Container not found.')

		# create a directory to store the files safely
		if not os.path.exists('hyperparameter_config_tmp'):
			os.mkdir('hyperparameter_config_tmp')
		os.chdir('hyperparameter_config_tmp')

		# write dict to json
		with open('hyperparameter_config.json', 'w') as config_json:
			config_json.write(json.dumps(hyperparameter_config_dict))

		# put config.json in tar archive
		with tarfile.open('hyperparameter_config.tar', 'w') as tar:
			try:
				tar.add('hyperparameter_config.json')
			finally:
				tar.close()

		# uploading the tar to the container
		ok = False
		with open('hyperparameter_config.tar', 'rb') as fd:
			ok = container.put_archive(path='/app', data=fd)

		# remove obsolete files
		if ok:
			os.remove('hyperparameter_config.json')
			os.remove('hyperparameter_config.tar')
			os.chdir('..')
			os.rmdir('hyperparameter_config_tmp')
		return DockerInfo(id=container_id, status=container.status, data=ok)

	def _initialize_port_mapping(cls):
		"""
		Initialize the cls._port_mapping dictionary.

		Opens the 'occupied_ports.txt' and reads the containers registered there, creating a dictionary of container_id:port mappings.
		Checks that the registered containers and the currently running containers are the same.
		"""
		# make sure the 'occupied_ports.txt' exists
		with open(os.path.join(os.path.dirname(__file__), 'occupied_ports.txt'), 'a'):
			pass
		# initialize the list of occupied ports by reading from the file
		with open(os.path.join(os.path.dirname(__file__), 'occupied_ports.txt'), 'r') as port_file:
			occupied_ports = port_file.readlines()
		# occupied_ports is a tuple with alternating container_id and port
		occupied_ports = (item[:-1] for item in occupied_ports)
		# the port mapping is a dictionary with the container_id being the key and its port the value
		cls._port_mapping = dict(zip(occupied_ports, occupied_ports))

		# make sure all containers are mapped and registered to the manager
		running_containers = [container.id for container in cls._client.containers.list(all=True)]
		mapped_containers = list(cls._port_mapping.keys())
		assert set(mapped_containers) == set(running_containers), \
f'''Container-Port mapping is mismatched! Check the \'occupied_ports.txt\' and your running docker containers (`docker ps -a`)!
running_containers: {running_containers}
mapped_containers: {mapped_containers}'''

		# make the ports integers
		cls._port_mapping.update((key, int(value)) for key, value in cls._port_mapping.items())

	# OBSERVER
	def attach(self, id: int, observer) -> None:
		"""
		Attach an observer to the container.
		"""
		observer.implements()
		self._observers[id] = observer

	def notify(self, message_id, message_text) -> None:
		"""
		Notify all observers about an event, events should be: the container is done, the container stopped working (any reason).
		The observer will implement observer.update(message_id, message_text)

		Args:
			message_id (int): The id of the event, so the system knows how to hande it.
			message_text (str): This is the message that will be displayed to the user.
		"""
		allowed_message_ids = [0, 1]
		assert message_id in allowed_message_ids, f'The message id is not allowed: {message_id}'

		for observer in self._observers:
			observer.update(message_id, message_text)
	# END OBSERVER


if __name__ == '__main__':  # pragma: no cover
	manager = DockerManager()
	for command in manager._allowed_commands:
		print(manager._confirm_image_exists(command, update=True), '\n')
