import docker
import os
import time


class AlphaBusinessDockerInfo():
	"""
	This class encapsules the return values for the rest api
	"""
	def __init__(self, container_id: int, is_alive: bool = None, data: str = None) -> None:
		self.id = container_id
		self.is_alive = is_alive
		self.data = data


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

		If an image with the provided name already exists, no new image will be built and the existing one will be returned.

		Args:
			imagename (str, optional): The name the image will have. Defaults to 'bp2021image'.
		"""
		# https://docker-py.readthedocs.io/en/stable/images.html
		# build image from dockerfile and name it accordingly
		print('Building image...')
		# Find out if an image with the name already exists and remove it afterwards
		try:
			old_img = self._client.images.get(imagename)
		except docker.errors.ImageNotFound:
			old_img = None
		img = self._client.images.build(path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), tag=imagename, forcerm=True)
		if old_img is not None and old_img.id != img[0].id:
			print('An image with this name already exists, it will be overwritten')
			self._client.images.remove(old_img.id[7:])
		# return id without the 'sha256:'-prefix
		return img[0].id[7:]

	def start_container(self, image_id: str, config={}) -> str:
		"""
		Start a container for the given image.

		Currently does not support loading a config file.

		Args:
			image_id (str): The id of the image to start the container for.
			config (str): a json containing parameters for the simulation

		Returns:
			str: The id of the started docker container
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		print('Starting container...')
		# name will be first tag without the ':latest'-postfix
		container_name = self._client.images.get(image_id).tags[0][:-7]
		# create a device request to use all available GPU devices with compute capabilities
		device_request_gpu = docker.types.DeviceRequest(driver='nvidia', count=-1, capabilities=[['compute']])
		container = self._client.containers.run(image_id, name=f'{container_name}_container', detach=True, network_mode='bridge', ports={'6006/tcp': 6006}, device_requests=[device_request_gpu])
		return container.id

	def container_status(self, container_id: str) -> str:
		"""
		Return the status of the given container.
		Will be one of 'restarting', 'running', 'paused', 'exited'.

		Args:
			container_id (str): The id of the container

		Returns:
			str: The status of the container
		"""
		return self._client.containers.get(container_id).status

	def is_container_running(self, container_id: str) -> bool:
		"""
		Returns `True` if the given container is still running and `False` otherwise.

		Args:
			container_id (str): The id of the container

		Returns:
			bool: Whether or not the container is still running
		"""
		return self.container_status(container_id) == 'running'

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

	def get_container_data(self, target_path: str, container_path: str, container_id: str) -> tuple:
		"""
		Return the data in the specified path. Will be returned as a tar archive.

		Args:
			target_path (str):
			path (str): [description]
			container_id (str): [description]

		Returns:
			tuple: [description]
		"""
		with open(target_path, 'wb') as f:
			bits, stats = self._client.containers.get(container_id).get_archive(path=container_path)
			for chunk in bits:
				f.write(chunk)

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

	def get_tensorboard_link(self, id: int) -> AlphaBusinessDockerInfo:
		"""
		This method should return the link to a running tensorboard of a container

		Args:
			id (int): id of running docker container

		Returns:
			str: link to tensorboard
		"""
		pass

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
	def container_progress(self, id: int) -> AlphaBusinessDockerInfo:
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
	cont = manager.start_container(img)
	print('Status:', manager.container_status(cont))
	print('Sleeping')
	time.sleep(5)
	print('Stdout of the container:\n')
	print(manager.get_container_logs(cont))
	print('Status:', manager.container_status(cont))
	time.sleep(3)
	print('Getting archive data...')
	manager.get_container_data(os.path.abspath(os.path.join(os.path.dirname(__file__), f'results_{cont[:10]}_{time.strftime("%b%d_%H-%M-%S")}.tar')), '/app/results', cont)
	print('Stopping container...')
	manager.stop_container(cont)
	print('Removing container...')
	manager.remove_container(cont)
