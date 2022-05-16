from abc import ABC, abstractmethod

from attrdict import AttrDict

from recommerce.market.sim_market import SimMarket
from recommerce.rl.callback import RecommerceCallback
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class RLTrainer(ABC):
	def __init__(self, marketplace_class: SimMarket, agent_class: ReinforcementLearningAgent, config_market: AttrDict, config_rl: AttrDict):
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
		self.config_market = config_market
		self.config_rl = config_rl
		assert self.trainer_agent_fit()

	def initialize_callback(self, training_steps):
		agent = self.agent_class(
			marketplace=self.marketplace_class(config=self.config_market),
			config_market=self.config_market,
			config_rl=self.config_rl)
		self.callback = RecommerceCallback(
			self.agent_class,
			self.marketplace_class,
			self.config_market,
			self.config_rl,
			training_steps,
			500,
			'dat',
			agent.name)
		self.callback.model = agent

	def consider_sync_tgt_net(self, frame_idx) -> None:
		if (frame_idx + 1) % self.config_rl.sync_target_frames == 0:
			self.callback.model.synchronize_tgt_net()

	@abstractmethod
	def train_agent(self, maxsteps) -> None:
		raise NotImplementedError('This method is abstract. Use a subclass')
