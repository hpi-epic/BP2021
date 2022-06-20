
from types import GeneratorType


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

	def __str__(self) -> str:
		return str(vars(self))
