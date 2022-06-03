from abc import ABC, abstractmethod

from attrdict import AttrDict


# This file contains all abstract vendors who are not made for a specific market situation (like circular and linear)
class Agent(ABC):

	def __init__(self, config_market: AttrDict, name=''):
		self.name = name if name != '' else type(self).__name__
		self.config_market = config_market

	@classmethod
	def custom_init(cls, class_name, args):
		"""
		Initialize an agent with a list of arguments.

		Args:
			class_name (agent class): The class of the agent that should be instantiated.
			args (list): List of arguments to pass the initializer.

		Returns:
			agent instance: An instance of the agent_class initialized with the given args.
		"""
		return class_name(*args) if len(args) > 0 else class_name()

	@abstractmethod
	def policy(self, observation, *_):  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')


class HumanPlayer(ABC):
	@abstractmethod
	def policy(self, observation, *_) -> int:  # pragma: no cover
		raise NotImplementedError('This method is abstract. Use a subclass')


class RuleBasedAgent(Agent, ABC):
	pass


class FixedPriceAgent(RuleBasedAgent, ABC):
	"""
	An abstract class for FixedPriceAgents
	"""
	pass
