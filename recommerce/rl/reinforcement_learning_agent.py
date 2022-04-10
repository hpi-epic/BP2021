from abc import ABC, abstractmethod

import torch

from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import Agent


class ReinforcementLearningAgent(Agent, ABC):
	@abstractmethod
	def __init__(
			self,
			marketplace: SimMarket,
			optim=None,
			device='cuda' if torch.cuda.is_available() else 'cpu',
			load_path=None,
			name='enter a name here'):
		"""
		Every ReinforcementLearningAgent must offer initialization by these parameters

		Args:
			marketplace (SimMarket): The marketplace the agent will interact with.
			load_path (str, optional): The path to load existing parameters of a network corresponding to this agent.
			Note that this only refers to a network responsible for behaviour.
			Assistance networks may be initialized differently.

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
