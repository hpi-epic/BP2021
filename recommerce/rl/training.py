from abc import ABC, abstractmethod

import torch

import recommerce.configuration.utils as ut
from recommerce.configuration.hyperparameter_config import config
from recommerce.rl.callback import RecommerceCallback
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class RLTrainer(ABC):
	def __init__(self, marketplace_class, agent_class):
		"""
		Initialize an RLTrainer to train one specific configuration.
		Args:
			marketplace_class (subclass of SimMarket): The market scenario you want to train.
			agent_class (subclass of RLAgent): The agent you want to train.
		"""
		# TODO: assert Agent and marketplace fit together
		assert issubclass(agent_class, ReinforcementLearningAgent)
		self.marketplace_class = marketplace_class
		self.agent_class = agent_class
		assert self.trainer_agent_fit()

	def initialize_callback(self, training_steps):
		agent = self.agent_class(marketplace=self.marketplace_class(), optim=torch.optim.Adam)
		self.callback = RecommerceCallback(self.agent_class, self.marketplace_class, training_steps, 500, 'dat', agent.name)
		self.callback.model = agent

	def calculate_dict_average(self, all_dicts) -> dict:
		"""
		Takes a list of dictionaries and calculates the average for each entry over all dicts.
		Assumes that all dicts have the same shape.
		Args:
			all_dicts (list of dicts): The dictionaries which entries you want to average
		Returns:
			dict: A dict of the same shape containing the average in each entry.
		"""
		sliced_dicts = all_dicts[-100:]
		averaged_info = sliced_dicts[0]
		for i, next_dict in enumerate(sliced_dicts):
			if i != 0:
				averaged_info = ut.add_content_of_two_dicts(averaged_info, next_dict)
		averaged_info = ut.divide_content_of_dict(averaged_info, len(sliced_dicts))
		return averaged_info

	def consider_sync_tgt_net(self, frame_idx) -> None:
		if (frame_idx + 1) % config.sync_target_frames == 0:
			self.callback.model.synchronize_tgt_net()

	@abstractmethod
	def train_agent(self, maxsteps) -> None:
		raise NotImplementedError('This method is abstract. Use a subclass')
