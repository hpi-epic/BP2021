from abc import ABC, abstractmethod


class Agent(ABC):

	def __init__(self, name='agent'):
		self.name = name

	def custom_init(class_name, args):
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


class ReinforcementLearningAgent(Agent, ABC):
	@abstractmethod
	def __init__(self, n_observations, n_actions, load_path=''):
		"""
		Every ReinforcementLearningAgent must offer initialization by these parameters

		Args:
			n_observations (int): length of input (observation) vector
			n_actions (int): length of output vector
			load_path (str, optional): The path to load existing parameters of a network corresponding to this agent.
			Note that this only refers to a network responsible for behaviour.
			Assistance networks may be initialized differently.
			Defaults to ''.

		Raises:
			NotImplementedError: This is an abstract interface definition
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def synchronize_tgt_net(self):
		"""
		This method writes the parameter from the value estimating net to it's target net.
		Call this method regularly during training.
		Having a target net solves problems occuring due to oscillation.
		"""
		raise NotImplementedError('This method is abstract. Use a subclass')
