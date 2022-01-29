import os
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
		return self._start_container(container.id, config)

	def health(self, container_id: str) -> DockerInfo:
		"""
		To call by the REST API. It provides the current status of the specified container.

		Args:
			container_id (str): The id of the container.

		Returns:
			DockerInfo: A JSON serializable object containing the id and the status of the new container.
		"""
		return DockerInfo(container_id, status=self._container_status(container_id))

	def start_tensorboard(self, container_id: str) -> str:
		"""
		Start a tensorboard in the specified container.
		Args:
			container_id (str): The id of the container.
		Returns:
			str: The link to the tensorboard session.
		"""
		# assert self.is_container_running(container_id), f'the Container is not running: {container_id}'
		self._execute_command(container_id, 'mkdir ./results/runs/')
		self._execute_command(container_id, 'tensorboard serve --logdir ./results/runs --bind_all')
		return DockerInfo(container_id, data='http://localhost:6006')

	def stop_container(self, container_id: str) -> bool:
		"""
		Stop a running container.

		Args:
			container_id (str): The id of the container.
		"""
		print('Stopping container', container_id)
		_ = self._client.containers.get(container_id).stop()
		# maybe the self.container_status(container_id).status) call can be replaced, beacuse we get a bool feedback from the docker api

		return DockerInfo(id=container_id, status=self._container_status(container_id))

	def remove_container(self, container_id: str) -> DockerInfo:
		"""
		Remove a stopped container.

		Will raise an error if the container is still running.

		Args:
			container_id (str): The id of the container.
		"""
		print('Removing container', container_id)
		container = self._client.containers.get(container_id)
		container.stop(timeout=20)
		container.remove()
		# this could fail, the status is not clear
		return DockerInfo(id=container_id, status='removed')

	# I would suggest an observer pattern for docker container:
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

	def _build_image(self, imagename: str = 'bp2021image') -> DockerInfo:
		"""
		Build an image from the default dockerfile and name it accordingly.

		If an image with the provided name already exists, that image will be overwritten.

		Args:
			imagename (str, optional): The name the image will have. Defaults to 'bp2021image'.

		Returns:
			DockerInfo: A DockerInfo object with id set.
		"""
		# https://docker-py.readthedocs.io/en/stable/images.html
		# build image from dockerfile and name it accordingly
		print('Building image...')
		# Find out if an image with the name already exists and remove it afterwards
		try:
			old_img = self._client.images.get(imagename)
		except docker.errors.ImageNotFound:
			old_img = None
		img, logs = self._client.images.build(path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
			tag=imagename, forcerm=True)
		if old_img is not None and old_img.id != img.id:
			print(f'An image with this name already exists, it will be overwritten: {imagename}')
			self._client.images.remove(old_img.id[7:])
		# return id without the 'sha256:'-prefix
		return DockerInfo(id=img.id[7:])

	def _create_container(self, image_id: str, use_gpu: bool = True) -> DockerInfo:
		"""
		Create a container for the given image.

		Args:
			image_id (str): The id of the image to create the container for.

		Returns:
			DockerInfo: A DockerInfo object with id and status set.
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		print('Creating container...')
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

	def _start_container(self, container_id: str, config: dict = {}) -> DockerInfo:
		"""
		Start a container for the given image.

		Currently does not support loading a config file.

		Args:
			container_id (str): The id of the image to start the container for.
			config (str): a json containing parameters for the simulation.

		Returns:
			str: The id of the started docker container.
		"""
		if self._container_status(container_id) == 'running':
			print(f'Container is already running: {container_id}')
			return DockerInfo(id=container_id, status='running')

		print('Starting container...')
		container = self._client.containers.get(container_id)
		container.start()
		return DockerInfo(id=container_id, status=self._container_status(container.id))

	def _execute_command(self, container_id: str, command: str) -> DockerInfo:
		print(f'Executing command: {command}')
		_, stream = self._client.containers.get(container_id).exec_run(cmd=command, stream=True)
		return DockerInfo(id=container_id, stream=stream)

	def _container_status(self, container_id) -> str:
		"""
		Return the status of the given container.
		Can e.g. be one of 'created', 'running', 'paused', 'exited' and more.

		Args:
			container_id (str): The id of the container

		Returns:
			str: The status of the container
		"""
		return self._client.containers.get(container_id).status

	# TODO: Refactor to return the raw data stream
	def get_container_data(self, container_path: str, container_id: str,
		target_filename: str = f'results_{time.strftime("%b%d_%H-%M-%S")}') -> DockerInfo:
		"""
		Save the data in the container_path to the target_path as a tar archive.

		Args:
			container_path (str): The path in the container from which to get the data.
			container_id (str): The id of the container.
			target_filename (str): The name of the target file. Defaults to 'results_{time.strftime("%b%d_%H-%M-%S")}'

		Returns:
			dict: The 'stats' metadata of the archive extraction.
		"""
		target_filename += '.tar'
		target_path = os.path.join(os.path.dirname(__file__), 'docker_archives', target_filename)
		with open(target_path, 'wb') as f:
			bits, stats = self._client.containers.get(container_id).get_archive(path=container_path)
			for chunk in bits:
				f.write(chunk)
			print(f'Archive written to: {os.path.abspath(target_path)}')
		return DockerInfo(container_id, data=stats)

	def _remove_image(self, image_id: str) -> None:
		"""
		Remove an image.

		Args:
			image_id (str): The id of the image.
		"""
		self._client.images.get(image_id).remove()

	def upload_file(self, container_id: str, src_path: str, dest_path: str) -> None:
		"""
		Upload a file to the specified container.

		Args:
			container_id (str): The id of the container.
			src_path (str): The path to the file to upload.
			dest_path (str): The path in the container to upload the file to.
		"""
		container = self._client.containers.get(container_id)
		container.copy()


if __name__ == '__main__':
	manager = DockerManager()
	img = manager._build_image()
	# print(img.id)
	# cont = manager.create_container(img.id, use_gpu=False)
	# info = manager.start_container(cont.id)
	# variables = vars(info)
	# print(variables)
	# json.dumps(variables)
	# tb_link = manager.start_tensorboard(cont.id)
	# print(f'Tensorboard started on: {tb_link}')
	# info = manager.execute_command(cont.id, 'python ./src/rl/training_scenario.py')
	# print('Getting archive data...')
	# manager.get_container_data('/app/results', cont.id)
	# print('Stopping container...')
	# manager.stop_container(cont)
	# print('Removing container...')
	# manager.remove_container(cont.id)
