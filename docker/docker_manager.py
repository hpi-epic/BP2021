import json
import os
import tarfile
import time

import docker


class DockerInfo():
	"""
	This class encapsules the return values for the rest api
	"""

	def __init__(self, id: str = None, status: str = None, data: str = None, stream=None) -> None:
		"""
		Args:
			id (str, optional): The sha256 id of the container.
			status (bool, optional): Status of the container. Returned by `container_status`.
			data (str, optional): Any other data, dependent on the function called this differs.
			stream (stream generator, optional): A stream generator.
		"""
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
	# This dictionary is very important. It contains a list of commands that users can send to a Docker container.
	allowed_commands = {
		'training': 'python ./src/rl/training_scenario.py',
		'exampleprinter': 'python ./src/monitoring/exampleprinter.py',
		'monitoring': 'python ./src/monitoring/agent_monitoring/am_monitoring.py',
		# for better readability, define commands users might want to perform above this comment and internal commands below
		'mkdirRuns': 'mkdir ./results/runs/',
		'tensorboard': 'tensorboard serve --logdir ./results/runs --bind_all'
	}

	def __new__(cls):
		"""
		This function makes sure that the `DockerManager` is a singleton.

		Returns:
			DockerManager: The DockerManager instance.
		"""
		if cls._instance is None:
			cls._instance = super(DockerManager, cls).__new__(cls)
			cls._client = docker.from_env()
		return cls._instance

	def start(self, config: dict) -> DockerInfo:
		"""
		To call by the REST API. It creates and starts a new docker cintainer from the default image.

		Args:
			config (dict): The config.json to replace the default one with.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		container = self._create_container('bp2021image', use_gpu=False)
		try:
			return self._start_container(container.id, config)
		except docker.errors.NotFound:
			return DockerInfo(id=container.id, status=f'container not found: {container.id}')

	def health(self, container_id: str) -> DockerInfo:
		"""
		To call by the REST API. It provides the current status of the specified container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		print(f'Checking health status for: {container_id}')
		try:
			return DockerInfo(container_id, status=self._container_status(container_id))
		except docker.errors.NotFound:
			return DockerInfo(id=container_id, status=f'container not found: {container_id}')

	async def execute_command(self, container_id: str, command_id: str) -> DockerInfo:
		"""
		Execute a command on the specified container.

		Args:
			container_id (str): The id of the container.
			command_id (str): The id of the command. Checked against self.allowed_commands.

		Returns:
			DockerInfo: A DockerInfo object containing the id of the container and a stream generator for the stdout of the command.
		"""
		if command_id not in self.allowed_commands:
			print(f'Command with ID {command_id} not allowed')
			return DockerInfo(id=container_id, status=f'command not allowed: {command_id}')

		command = self.allowed_commands[command_id]
		print(f'Executing command: {command}')
		_, stream = self._client.containers.get(container_id).exec_run(cmd=command, stream=True)
		return DockerInfo(id=container_id, status=self._container_status(container_id), stream=stream)

	async def start_tensorboard(self, container_id: str) -> DockerInfo:
		"""
		Start a tensorboard in the specified container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A DockerInfo object containing the id of the container and a link to the tensorboard in the data field.
		"""
		await self.execute_command(container_id, 'mkdirRuns')
		await self.execute_command(container_id, 'tensorboard')
		return DockerInfo(container_id, data='http://localhost:6006')

	def get_container_data(self, container_id: str, container_path: str) -> DockerInfo:
		"""
		Return a data stream object that matches a .tar archive as well as a filename derived from the curent time and file-path.

		Args:
			container_id (str): The id of the container.
			container_path (str): The path in the container from which to get the data. Defaults to '/app/results'.

		Returns:
			DockerInfo: Contains the container_id, a filename in the data field and a stream generator matching the .tar archive.
		"""
		bits, _ = self._client.containers.get(container_id).get_archive(path=container_path)
		return DockerInfo(container_id, data=f'archive_{container_path.rpartition("/")[2]}_{time.strftime("%b%d_%H-%M-%S")}', stream=bits)

	def stop_container(self, container_id: str) -> DockerInfo:
		"""
		Stop a running container.

		After 10 seconds of no response, the container will be killed.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the container.
		"""
		print(f'Stopping container: {container_id}')
		self._client.containers.get(container_id).stop(timeout=10)

		return DockerInfo(id=container_id, status=self._container_status(container_id))

	def remove_container(self, container_id: str) -> DockerInfo:
		"""
		Remove and stop a container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the container.
		"""
		print(f'Removing container: {container_id}')
		try:
			self.stop_container(container_id)
			self._client.containers.get(container_id).remove()
			return DockerInfo(id=container_id, status='removed')
		except docker.errors.NotFound:
			return DockerInfo(container_id, status=f'container not found: {container_id}')

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
			message_id (int): The id of the event., so the system knows how to hande it.
			message_text (str): This is the message that will be displayed to the user.
		"""
		allowed_message_ids = [0, 1]
		assert message_id in allowed_message_ids, f'The message id is not allowed: {message_id}'

		for observer in self._observers:
			observer.update(message_id, message_text)

	def _build_image(self, imagename: str = 'bp2021image') -> str:
		"""
		Build an image from the default dockerfile and name it accordingly.

		If an image with the provided name already exists, that image will be overwritten.

		Args:
			imagename (str, optional): The name the image will have. Defaults to 'bp2021image'.

		Returns:
			str: The id of the image.
		"""
		# https://docker-py.readthedocs.io/en/stable/images.html
		# build image from dockerfile and name it accordingly
		print(f'Building image: {imagename}')
		# Find out if an image with the name already exists and remove it afterwards
		try:
			old_img = self._client.images.get(imagename)
		except docker.errors.ImageNotFound:
			old_img = None
		img, _ = self._client.images.build(path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
			tag=imagename, forcerm=True)
		if old_img is not None and old_img.id != img.id:
			print(f'An image with this name already exists, it will be overwritten: {imagename}')
			self._client.images.remove(old_img.id[7:])
		# return id without the 'sha256:'-prefix
		return img.id[7:]

	def _create_container(self, image_id: str, use_gpu: bool = True) -> DockerInfo:
		"""
		Create a container for the given image.

		Args:
			image_id (str): The id of the image to create the container for.

		Returns:
			DockerInfo: A DockerInfo object with id and status set.
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		print(f'Creating container for image: {image_id}')
		# name will be first tag without the ':latest'-postfix
		container_name = self._client.images.get(image_id).tags[0][:-7]
		# create a device request to use all available GPU devices with compute capabilities
		if use_gpu:
			device_request_gpu = docker.types.DeviceRequest(driver='nvidia', count=-1, capabilities=[['compute']])
			container = self._client.containers.create(image_id,
				name=f'{container_name}_container',
				detach=True,
				ports={'6006/tcp': 6006},
				device_requests=[device_request_gpu])
		else:
			container = self._client.containers.create(image_id, name=f'{container_name}_container', detach=True, ports={'6006/tcp': 6006})

		return DockerInfo(id=container.id, status=self._container_status(container.id))

	def _start_container(self, container_id: str, config: dict) -> DockerInfo:
		"""
		Start a container for the given image.

		Args:
			container_id (str): The id of the image to start the container for.
			config (str): a json containing parameters for the simulation.

		Returns:
			DockerInfo: A DockerInfo object with the id, the status of the started docker container and a bool in the data field
				indicating whether or not the config was uploaded successfully.
		"""
		if self._container_status(container_id) == 'running':
			print(f'Container is already running: {container_id}')
			return DockerInfo(id=container_id, status='running')

		print(f'Starting container: {container_id}')
		container = self._client.containers.get(container_id)
		container.start()
		upload_status = self._upload_config(container_id, config).data
		if not upload_status:
			print('Failed to upload configuration file!')
		return DockerInfo(id=container_id, status=self._container_status(container.id), data=upload_status)

	def _container_status(self, container_id) -> str:
		"""
		Return the status of the given container.

		Can e.g. be 'created', 'running', 'paused', 'exited'.

		Args:
			container_id (str): The id of the container.

		Returns:
			str: The status of the container.
		"""
		return self._client.containers.get(container_id).status

	def _remove_image(self, image_id: str) -> None:
		"""
		Remove an image.

		Args:
			image_id (str): The id of the image.
		"""
		self._client.images.get(image_id).remove()

	def _upload_config(self, container_id: str, config_dict: dict) -> DockerInfo:
		"""
		Upload a file to the specified container.

		Args:
			container_id (str): The id of the container.
			config_dict (dict): The config dictionary to upload.

		Returns:
			DockerInfo: A DockerInfo object with the id of the container and the status of the upload in the data field.
		"""
		# create a directory to store the files safely
		if not os.path.exists('config_tmp'):
			os.makedirs('config_tmp')
		os.chdir('config_tmp')
		container = self._client.containers.get(container_id)

		# write dict to json
		with open('config.json', 'w') as config_json:
			config_json.write(json.dumps(config_dict))

		# put config.json in tar archive
		with tarfile.open('config.tar', 'w') as tar:
			try:
				tar.add('config.json')
			finally:
				tar.close()

		# uploading the tar to the container
		ok = False
		with open('config.tar', 'rb') as fd:
			ok = container.put_archive(path='/app', data=fd)

		# remove obsolete files
		if ok:
			os.remove('config.json')
			os.remove('config.tar')
		return DockerInfo(id=container_id, data=ok)


if __name__ == '__main__':
	manager = DockerManager()
	img = manager._build_image()
