import docker
import os
import time


class DockerInfo():
	"""
	This class encapsules the return values for the rest api
	"""
	def __init__(self, container_id: int, status: bool = None, stream=None, data: str = None, info: str = None) -> None:
		self.id = container_id
		self.status = status
		self.stream = stream
		self.data = data
		self.info = info


class DockerManager():
	_instance = None
	_client = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(DockerManager, cls).__new__(cls)
			cls._client = docker.from_env()
		return cls._instance

	def build_image(self, imagename: str = 'bp2021image') -> str:
		"""
		Build an image from the default dockerfile and name it accordingly.

		If an image with the provided name already exists, that image will be overwritten.

		Args:
			imagename (str, optional): The name the image will have. Defaults to 'bp2021image'.

		Returns:
			DockerInfo: A DockerInfo object with id and stream set
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
		return DockerInfo(id=img.id[7:], stream=logs)

	def create_container(self, image_id: str) -> str:
		"""
		Create a container for the given image.

		Args:
			image_id (str): The id of the image to start the container for.

		Returns:
			str: The id of the created docker container
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		print('Creating container...')
		# name will be first tag without the ':latest'-postfix
		container_name = self._client.images.get(image_id).tags[0][:-7]
		# create a device request to use all available GPU devices with compute capabilities
		device_request_gpu = docker.types.DeviceRequest(driver='nvidia', count=-1, capabilities=[['compute']])
		container = self._client.containers.create(image_id, name=f'{container_name}_container', detach=True, ports={'6006/tcp': 6006},
			device_requests=[device_request_gpu])
		return container.id

	def start_container(self, container_id: str, config: dict = {}) -> str:
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
			return container_id
		print('Starting container...')
		self._client.containers.get(container_id).start()
		return container_id

	def execute_command(self, container_id: str, command: str):
		print(f'Executing command: {command}')
		_, stream = self._client.containers.get(container_id).exec_run(cmd=command, stream=True)
		return stream

	def container_status(self, container_id: str) -> str:
		"""
		Return the status of the given container.
		Can e.g. be one of 'created', 'running', 'paused', 'exited' and more.

		Args:
			container_id (str): The id of the container

		Returns:
			str: The status of the container
		"""
		return self._client.containers.get(container_id).status

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
		return self._client.containers.get(container_id).logs(timestamps=timestamps).decode('UTF-8')

	# TODO: Refactor to return the raw data stream
	def get_container_data(self, container_path: str, container_id: str,
		target_filename: str = f'results_{time.strftime("%b%d_%H-%M-%S")}') -> dict:
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
		return stats

	def stop_container(self, container_id: str) -> bool:
		"""
		Stop a running container.

		Args:
			container_id (str): The id of the container.
		"""
		return self._client.containers.get(container_id).stop()

	def remove_container(self, container_id: str) -> None:
		"""
		Remove a stopped container.

		Will raise an error if the container is still running.

		Args:
			container_id (str): The id of the container.
		"""
		self._client.containers.get(container_id).remove()

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
		assert self.is_container_running(container_id), f'the Container is not running: {container_id}'
		self.execute_command(container_id, 'mkdir ./results/runs/')
		self.execute_command(container_id, 'tensorboard serve --logdir ./results/runs --bind_all')
		return 'http://localhost:6006'

	# I would suggest an observer pattern for docker container:
	def attach(self, id: int, observer) -> None:
		"""
		Attach an observer to the container.
		"""
		pass

	def notify(self) -> None:
		"""
		Notify all observers about an event, events should be: the container is done, the container stopped working (any reason).
		The observer will implement observer.update(msg)
		"""
		pass

	# optional methods
	def container_progress(self, id: int) -> DockerInfo:
		"""
		This methoud should return the total progress of the container

		Args:
			id (int): id of running docker container

		Returns:
			int: progress between 0 and 1, 1 means done
		"""
		pass


if __name__ == '__main__':
	manager = DockerManager()
	img = manager.build_image()
	cont = manager.create_container(img)
	manager.start_container(cont)
	tb_link = manager.start_tensorboard(cont)
	print(f'Tensorboard started on: {tb_link}')
	stream = manager.execute_command(cont, 'python ./src/rl/training_scenario.py')
	print()
	for data in stream:
		print(data.decode(), end='')
	print()
	print('Getting archive data...')
	manager.get_container_data('/app/results', cont)
	print('Stopping container...')
	manager.stop_container(cont)
	print('Removing container...')
	manager.remove_container(cont)
