from random import randrange

class AlphaBusinessDockerInfo():
	"""This class encapsules the return values for the rest api
	"""
	def __init__(self, container_id: int, is_alive: str = None, data: str = None) -> None:
		self.id = container_id
		self.health_status = is_alive
		self.data = data


class AlphaBusinessDockerManager():
	# important methods:
	def start_docker(self, config) -> AlphaBusinessDockerInfo:
		"""
		This method should start a docker container with the given `config` as parameter configuration.

		Args:
			config (str): a json containing parameters for the simulation

		Returns:
			int: The id of the started docker container
		"""
		return AlphaBusinessDockerInfo(container_id=randrange(100))

	def is_container_alive(self, id: int) -> AlphaBusinessDockerInfo:
		"""
		This method should tell me if the docker container with the given id is still running.
		If nothing can destroy the observer notification (see below) we can remove this function.

		Args:
			id (int): id of running docker container

		Returns:
			bool: answers if the docker container with the id is running
		"""
		return AlphaBusinessDockerInfo(container_id=id, is_alive='very well')

	def get_container_data(self, id: int) -> AlphaBusinessDockerInfo:
		"""
		This method should return all data the docker container with a given id has produced yet.
		We should think about wrapping this data into an AlphaBusinessDataClass in order to return files etc.

		Args:
			id (int): id of running docker container

		Returns:
			str: produced data
		"""
		return AlphaBusinessDockerInfo(container_id=id, data='this is a test')

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
