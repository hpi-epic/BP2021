from abc import abstractmethod

import torch
from attrdict import AttrDict

from recommerce.configuration.json_configurable import JSONConfigurable
from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import Agent


class ReinforcementLearningAgent(Agent, JSONConfigurable):
	@abstractmethod
	def __init__(
			self,
			config_market: AttrDict,
			config_rl: AttrDict,
			marketplace: SimMarket,
			device='cuda' if torch.cuda.is_available() else 'cpu',
			load_path=None,
			name='ReinforcementLearningAgent'):
		"""
		Every ReinforcementLearningAgent must offer initialization by these parameters

		Args:
			marketplace (SimMarket): The marketplace the agent will interact with.
			device (str): The device the agent will be trained on.
			load_path (str, optional): The path to load existing parameters of a network corresponding to this agent.
			Note that this only refers to a network responsible for behaviour.
			Assistance networks may be initialized differently.
			name (str, optional): The name of the agent. Defaults to 'ReinforcementLearningAgent'.

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
