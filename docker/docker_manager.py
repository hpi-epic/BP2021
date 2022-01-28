import os
import time
import json
import docker


class DockerInfo():
	"""
	This class encapsules the return values for the rest api
	"""
	def __init__(self, id: str = "", status: str = "", data: str = "") -> None:
		"""
		Args:
			id (str, optional): The sha256 id of the object.
			type (str, optional): Will be one of 'image', 'container'.
			status (bool, optional): Status of the container. Returned by `container_status`.
			stream ([type], optional): Will be a stream generator object. Returned by `build_image`, `execute_command`.
			data (str, optional): Raw string output that can be printed as is. Returned by `get_container_logs`.
		"""
		self.id = id
		self.status = status
		self.data = data


class DockerManager():
	_instance = None
	_client = None
	_observers = []

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(DockerManager, cls).__new__(cls)
			cls._client = docker.from_env()
		return cls._instance

	def build_image(self, imagename: str = 'bp2021image') -> DockerInfo:
		"""
		Build an image from the default dockerfile and name it accordingly.

		If an image with the provided name already exists, that image will be overwritten.

		Args:
			imagename (str, optional): The name the image will have. Defaults to 'bp2021image'.

		Returns:
			DockerInfo: A DockerInfo object with id, type and stream set.
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

	def create_container(self, image_id: str, use_gpu: bool = True) -> DockerInfo:
		"""
		Create a container for the given image.

		Args:
			image_id (str): The id of the image to create the container for.

		Returns:
			DockerInfo: A DockerInfo object with id, type and status set.
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

		return DockerInfo(id=container.id, status=self.container_status(container.id))

	def start_container(self, container_id: str, config: dict = {}) -> DockerInfo:
		"""
		Start a container for the given image.

		Currently does not support loading a config file.

		Args:
			container_id (str): The id of the image to start the container for.
			config (str): a json containing parameters for the simulation.

		Returns:
			str: The id of the started docker container.
		"""
		if self.container_status(container_id) == 'running':
			print(f'Container is already running: {container_id}')
			return DockerInfo(id=container_id, status=self.container.status)
		print('Starting container...')
		container = self._client.containers.get(container_id)
		container.start()
		return DockerInfo(id=container_id, status=self.container_status(container.id))

	def execute_command(self, container_id: str, command: str) -> DockerInfo:
		print(f'Executing command: {command}')
		_, stream = self._client.containers.get(container_id).exec_run(cmd=command, stream=True)
		return DockerInfo(id=container_id)

	def container_status(self, container_id) -> str:
		"""
		Return the status of the given container.
		Can e.g. be one of 'created', 'running', 'paused', 'exited' and more.

		Args:
			container_id (str): The id of the container

		Returns:
			str: The status of the container
		"""
		return DockerInfo(id=container_id, status=self._client.containers.get(container_id).status)

	def get_container_logs(self, container_id: str, timestamps: bool = False) -> str:
		"""
		Return the logs of the given container.

		The logs consist of `STDOUT` and `STDERR`.

		Args:
			container_id (str): The id of the container.
			timestamps (bool): Whether or not timestamps should be displayed in the logs.

		Returns:
			str: The logs of the container
		"""
		return DockerInfo(container_id, type='container',
			data=self._client.containers.get(container_id).logs(timestamps=timestamps).decode('UTF-8'))

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
		return DockerInfo(stats)

	def stop_container(self, container_id: str) -> bool:
		"""
		Stop a running container.

		Args:
			container_id (str): The id of the container.
		"""
		print('Stopping container', container_id)
		_ = self._client.containers.get(container_id).stop()
		# maybe the self.container_status(container_id).status) call can be replaced, beacuse we get a bool feedback from the docker api
		return DockerInfo(id=container_id, status=self.container_status(container_id).status)

	def remove_container(self, container_id: str) -> DockerInfo:
		"""
		Remove a stopped container.

		Will raise an error if the container is still running.

		Args:
			container_id (str): The id of the container.
		"""
		print('Removing container', container_id)
		self._client.containers.get(container_id).remove()
		# this could fail, the status is not clear
		return DockerInfo(id=container_id, status=self.container_status(container_id).status)

	def remove_image(self, image_id: str) -> None:
		"""
		Remove an image.

		Args:
			image_id (str): The id of the image.
		"""
		self._client.images.get(image_id).remove()

	def start_tensorboard(self, container_id: str) -> str:
		"""
		Start a tensorboard in the specified container.

		Args:
			container_id (str): The id of the container.

		Returns:
			str: The link to the tensorboard session.
		"""
		# assert self.is_container_running(container_id), f'the Container is not running: {container_id}'
		self.execute_command(container_id, 'mkdir ./results/runs/')
		self.execute_command(container_id, 'tensorboard serve --logdir ./results/runs --bind_all')
		return 'http://localhost:6006'

	# def upload_file(self, container_id:str, src_path: str, dest_path: str) -> None:
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

# -----------------------------------------------------------------------------------------------------
	def start(self, config: dict) -> DockerInfo:
		img = self.build_image()
		cont = self.create_container(img.id, use_gpu=False)
		return self.start_container(cont.id, config)

if __name__ == '__main__':
	manager = DockerManager()
	img = manager.build_image()
	print(img.id)
	cont = manager.create_container(img.id, use_gpu=False)
	info = manager.start_container(cont.id)
	json.dumps(info)
	tb_link = manager.start_tensorboard(cont.id)
	print(f'Tensorboard started on: {tb_link}')
	info = manager.execute_command(cont.id, 'python ./src/rl/training_scenario.py')
	print('Getting archive data...')
	manager.get_container_data('/app/results', cont.id)
	print('Stopping container...')
	manager.stop_container(cont)
	print('Removing container...')
	manager.remove_container(cont.id)
