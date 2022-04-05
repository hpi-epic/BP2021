from abc import ABC, abstractmethod

from recommerce.market.vendors import Agent


class ReinforcementLearningAgent(Agent, ABC):
	@abstractmethod
	def __init__(
			self,
			n_observations=None,
			n_actions=None,
			marketplace=None,
			optim=None,
			device='cuda' if torch.cuda.is_available() else 'cpu',
			load_path=None,
			name='enter a name here'):
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
