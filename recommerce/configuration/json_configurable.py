from abc import ABC, abstractmethod


class JSONConfigurable(ABC):
	@staticmethod
	@abstractmethod
	def get_configurable_fields():
		raise NotImplementedError
