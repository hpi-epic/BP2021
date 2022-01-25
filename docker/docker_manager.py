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


class AlphaBusinessDockerManager():

	def __init__(self):
		self.client = docker.from_env(version='1.40')

	def build_image(self, imagename: str = 'bp2021image'):
		"""
		Build an image from the default dockerfile and name it accordingly.

		If an image with the provided name already exists, no new image will be built an the existing one will be returned.

		Args:
			imagename (str, optional): The name the image will have. Defaults to 'bp2021image'.
		"""
		# https://docker-py.readthedocs.io/en/stable/images.html
		# build image from dockerfile and name it accordingly
		img = self.client.images.build(path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), tag=imagename, forcerm=True)
		# return id without the 'sha256:'-prefix
		return img[0].id[7:]

	def start_container(self, image: str, config={}) -> AlphaBusinessDockerInfo:
		"""
		This method should start a docker container with the given `config` as parameter configuration.

		Args:
			config (str): a json containing parameters for the simulation

		Returns:
			int: The id of the started docker container
		"""
		# https://docker-py.readthedocs.io/en/stable/containers.html
		# options: auto_remove, command, detach, environment (dict/list of environment variables to set), healthcheck(?), name, log_config, remove
		# restart_policy: Restart the container when it exits. Configured as a dictionary with keys: Name One of on-failure, or always.
		# MaximumRetryCount Number of times to restart the container on failure. For example: {"Name": "on-failure", "MaximumRetryCount": 5}
		print('Starting container...')
		# name will be first tag without the ':latest'-postfix
		container_name = self.client.images.get(image).tags[0][:-7]
		# create a device request to use all available GPU devices with compute capabilities
		device_request_gpu = docker.types.DeviceRequest(driver='nvidia', count=1, capabilities=[['compute']])
		container = self.client.containers.run(image, name=f'{container_name}_container', detach=True, device_requests=[device_request_gpu])
		return container.id

	# formerly is_container_alive
	def container_status(self, container_id: str) -> AlphaBusinessDockerInfo:
		"""
		This method should tell me if the docker container with the given id is still running.
		If nothing can destroy the observer notification (see below) we can remove this function.

		Args:
			id (int): id of running docker container

		Returns:
			bool: answers if the docker container with the id is running
		"""
		return self.client.containers.get(container_id).status

	def get_container_data(self, container_id: str) -> AlphaBusinessDockerInfo:
		"""
		This method should return all data the docker container with a given id has produced yet.
		We should think about wrapping this data into an AlphaBusinessDataClass in order to return files etc.

		Args:
			id (int): id of running docker container

		Returns:
			str: produced data
		"""
		return manager.client.containers.get(container_id).logs().decode('UTF-8')

	def kill_container(self, id: int) -> None:
		"""
		kills a docker container with a given id

		Args:
			id (int): id of docker conrainer
		"""
		pass

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
	manager = AlphaBusinessDockerManager()
	img = manager.build_image()
	cont = manager.start_container(img)
	print('Status:', manager.container_status(cont))
	time.sleep(3)
	print('Stdout of the container:\n')
	print(manager.get_container_data(cont))
	print('Status:', manager.container_status(cont))
