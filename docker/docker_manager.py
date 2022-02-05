import json
import os
import tarfile
import time

import docker
from docker.models.containers import Container


class DockerInfo():
	"""
	This class encapsules the return values for the REST API.
	"""

	def __init__(self, id: str, status: str, data: str = None, stream=None) -> None:
		"""
		Args:
			id (str, optional): The sha256 id of the container. Always returned.
			status (bool, optional): Status of the container. Always returned.
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
	_allowed_commands = {
		'training': './src/rl/training_scenario.py',
		'exampleprinter': './src/monitoring/exampleprinter.py',
		'monitoring': './src/monitoring/agent_monitoring/am_monitoring.py'
	}
	counter = 6006

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

	def start(self, command_id: str, config: dict) -> DockerInfo:
		"""
		To be called by the REST API. Create and start a new docker container from the default image.

		Args:
			config (dict): The config.json to replace the default one with.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		if command_id not in self._allowed_commands:
			print(f'Command with ID {command_id} not allowed')
			return DockerInfo(id=None, status=f'Command not allowed: {command_id}')

		# Hacky dockerfile creation
		with open('../dockerfile', 'r') as dockerfile:
			template = dockerfile.read()
			dock = template
			dock = dock.replace('USER_COMMAND_PLACEHOLDER', self._allowed_commands[command_id])
		with open('../dockerfile', 'w') as dockerfile:
			dockerfile.write(dock)
		try:
			self._build_image(imagename='bp2021image')
		finally:
			with open('../dockerfile', 'w') as dockerfile:
				dockerfile.write(template)

		container_info = self._create_container('bp2021image', config, use_gpu=False)

		if container_info.status.__contains__('Container not found') or container_info.data is False:
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
			return DockerInfo(container_id, status=f'Container not found: {container_id}')
		return DockerInfo(container_id, status=container.status)

	# TODO:
	# When we add the possibility to run multiple containers at once, we need to change the port that is exposed in each container,
	# 	which also leads to a different port in the return value here.
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
			return DockerInfo(container_id, status=f'Container not found: {container_id}')

		container.exec_run(cmd='tensorboard serve --logdir ./results/runs --bind_all')
		return DockerInfo(container_id, status=container.status, data='http://localhost:6006')

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
			return DockerInfo(container_id, status=f'Container not found: {container_id}')

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
			return DockerInfo(container_id, status=f'Container not found: {container_id}')

		bits, _ = container.get_archive(path=container_path)
		return DockerInfo(container_id, data=f'archive_{container_path.rpartition("/")[2]}_{time.strftime("%b%d_%H-%M-%S")}', stream=bits)

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
			return DockerInfo(container_id, status=f'Container not found: {container_id}')

		self._stop_container(container_id)

		print(f'Removing container: {container_id}')
		try:
			container.remove()
			return DockerInfo(id=container_id, status='removed')
		except docker.errors.APIError:
			return DockerInfo(id=container_id, status=f'APIError encountered while removing container: {container_id}')

	# PRIVATE METHODS
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

	def _create_container(self, image_id: str, config: dict, use_gpu: bool = True) -> DockerInfo:
		"""
		Create a container for the given image.

		Args:
			image_id (str): The id of the image to create the container for.
			config (str): A json containing parameters for the simulation.
			use_gpu (bool) Whether or not to request access to GPU's. Defaults to True.

		Returns:
			DockerInfo: A DockerInfo object with id and status set.
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		print(f'Creating container for image: {image_id}')
		# name will be first tag without the ':latest'-postfix
		# container_name = self._client.images.get(image_id).tags[0][:-7]

		# create a device request to use all available GPU devices with compute capabilities
		if use_gpu:
			device_request_gpu = docker.types.DeviceRequest(driver='nvidia', count=-1, capabilities=[['compute']])
			container: Container = self._client.containers.create(image_id,
				# name=f'{container_name}_container',
				detach=True,
				ports={'6006/tcp': self.counter},
				device_requests=[device_request_gpu])
		else:
			try:
				container: Container = self._client.containers.create(image_id, detach=True, ports={'6006/tcp': self.counter})
			except docker.errors.ImageNotFound:
				return DockerInfo(id=image_id, status=f'Image not found: {image_id}')

		self.counter += 1

		upload_info = self._upload_config(container.id, config)
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
			return DockerInfo(id=container_id, status=f'Container not found: {container_id}')

		if container.status == 'running':
			print(f'Container is already running: {container_id}')
			return DockerInfo(id=container_id, status='running')

		print(f'Starting container: {container_id}')
		try:
			container.start()
			return DockerInfo(id=container_id, status=container.status)
		except docker.errors.APIError:
			return DockerInfo(id=container_id, status=f'APIError encountered while starting container: {container_id}')

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

	# Should no longer be used since we moved command execution to the container ENTRYPOINT
	def _execute_command(self, container_id: str, command_id: str) -> DockerInfo:
		"""
		Execute a command on the specified container.

		Args:
			container_id (str): The id of the container.
			command_id (str): The id of the command. Checked against self._allowed_commands.

		Returns:
			DockerInfo: A DockerInfo object containing the id and status of the container.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(container_id, status=f'Container not found: {container_id}')

		if command_id not in self._allowed_commands:
			print(f'Command with ID {command_id} not allowed')
			self.remove_container(container_id)
			return DockerInfo(id=container_id, status=f'Command not allowed: {command_id}')

		command = self._allowed_commands[command_id]
		print(f'Executing command: {command}')
		_, stream = container.exec_run(cmd=command, detach=True)
		return DockerInfo(id=container_id, status=container.status)

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
			return DockerInfo(container_id, status=f'Container not found: {container_id}')

		print(f'Stopping container: {container_id}')
		try:
			container.stop(timeout=10)
			return DockerInfo(id=container_id, status=container.status)
		except docker.errors.APIError:
			return DockerInfo(container_id, status=f'APIError encountered while stopping container: {container_id}')

	def _upload_config(self, container_id: str, config_dict: dict) -> DockerInfo:
		"""
		Upload a file to the specified container.

		Args:
			container_id (str): The id of the container.
			config_dict (dict): The config dictionary to upload.

		Returns:
			DockerInfo: A DockerInfo object with the id of the container and the status of the upload in the data field.
		"""
		container: Container = self._get_container(container_id)
		if not container:
			return DockerInfo(id=container_id, status=f'Container not found: {container_id}')

		# create a directory to store the files safely
		if not os.path.exists('config_tmp'):
			os.mkdir('config_tmp')
		os.chdir('config_tmp')

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
			os.chdir('..')
			os.rmdir('config_tmp')
		return DockerInfo(id=container_id, status=container.status, data=ok)

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
			message_id (int): The id of the event., so the system knows how to hande it.
			message_text (str): This is the message that will be displayed to the user.
		"""
		allowed_message_ids = [0, 1]
		assert message_id in allowed_message_ids, f'The message id is not allowed: {message_id}'

		for observer in self._observers:
			observer.update(message_id, message_text)
# END OBSERVER


if __name__ == '__main__':
	manager = DockerManager()
	img = manager._build_image()
