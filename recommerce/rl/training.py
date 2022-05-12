from abc import ABC, abstractmethod

from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.market.sim_market import SimMarket
from recommerce.rl.callback import RecommerceCallback
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class RLTrainer(ABC):
	def __init__(
		self,
		marketplace_class: SimMarket,
		agent_class: ReinforcementLearningAgent,
		config: HyperparameterConfig,
		competitors: list = None):
		"""
		Initialize an RLTrainer to train one specific configuration.

		Args:
			marketplace_class (subclass of SimMarket): The market scenario you want to train.
			agent_class (subclass of RLAgent): The agent you want to train.
			competitors (list | None, optional): If set, which competitors should be used instead of the default ones.
		"""
		# TODO: assert Agent and marketplace fit together
		assert issubclass(agent_class, ReinforcementLearningAgent)
		self.marketplace_class = marketplace_class
		self.agent_class = agent_class
		self.competitors = competitors
		self.config = config
		assert self.trainer_agent_fit()

	def initialize_callback(self, training_steps):
		# This marketplace gets returned
		marketplace = self.marketplace_class(config=self.config, competitors=self.competitors)
		agent = self.agent_class(marketplace=marketplace, config=self.config)
		self.callback = RecommerceCallback(self.agent_class, self.marketplace_class, self.config, training_steps, 500, 'dat', agent.name)
		self.callback.model = agent
		return marketplace

	def consider_sync_tgt_net(self, frame_idx) -> None:
		if (frame_idx + 1) % self.config.sync_target_frames == 0:
			self.callback.model.synchronize_tgt_net()

	@abstractmethod
	def train_agent(self, maxsteps) -> None:
		raise NotImplementedError('This method is abstract. Use a subclass')
