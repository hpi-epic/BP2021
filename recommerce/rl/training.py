from abc import ABC, abstractmethod

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
		agent = self.agent_class(marketplace=self.marketplace_class())
		self.callback = RecommerceCallback(self.agent_class, self.marketplace_class, training_steps, 500, 'dat', agent.name)
		self.callback.model = agent

	def consider_sync_tgt_net(self, frame_idx) -> None:
		if (frame_idx + 1) % config.sync_target_frames == 0:
			self.callback.model.synchronize_tgt_net()

	@abstractmethod
	def train_agent(self, maxsteps) -> None:
		raise NotImplementedError('This method is abstract. Use a subclass')
