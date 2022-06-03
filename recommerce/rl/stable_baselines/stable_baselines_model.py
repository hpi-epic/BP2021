import os
from abc import ABC, abstractmethod

import numpy as np
from attrdict import AttrDict
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.market.sim_market import SimMarket
from recommerce.rl.callback import RecommerceCallback
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class StableBaselinesAgent(ReinforcementLearningAgent, LinearAgent, CircularAgent, ABC):
	def __init__(self, config_market: AttrDict, config_rl: AttrDict, marketplace, load_path=None, name=''):
		assert marketplace is not None
		assert isinstance(marketplace, SimMarket), \
			f'if marketplace is provided, marketplace must be a SimMarket, but is {type(marketplace)}'
		assert load_path is None or isinstance(load_path, str)
		assert name is None or isinstance(name, str)
		self.config_market = config_market
		self.config_rl = config_rl
		self.tensorboard_log = os.path.join(PathManager.results_path, 'runs')
		self.marketplace = marketplace
		if load_path is None:
			self._initialize_model(marketplace)
			print(f'I initiate {self.name}-agent using {self.model.device} device')
		if load_path is not None:
			self._load(load_path)
			print(f'I load {self.name}-agent using {self.model.device} device from {load_path}')

		self.name = name if name != '' else type(self).__name__

	@abstractmethod
	def _initialize_model(self, marketplace):
		raise NotImplementedError('This method is abstract. Use a subclass')

	@abstractmethod
	def _load(self, load_path):
		raise NotImplementedError('This method is abstract. Use a subclass')

	def policy(self, observation: np.array) -> np.array:
		assert isinstance(observation, np.ndarray), f'{observation}: this is a {type(observation)}, not a np ndarray'
		return self.model.predict(observation)[0]

	def synchronize_tgt_net(self):  # pragma: no cover
		assert False, 'This method may never be used in a StableBaselinesAgent!'

	def set_marketplace(self, new_marketplace: SimMarket):
		self.marketplace = new_marketplace
		self.model.set_env(new_marketplace)

	def train_agent(self, training_steps=100000, iteration_length=500, analyze_after_training=True):
		callback = RecommerceCallback(
			type(self), type(self.marketplace), self.config_market, self.config_rl, training_steps=training_steps, iteration_length=iteration_length,
			signature=self.name, analyze_after_training=analyze_after_training)
		self.model.learn(training_steps, callback=callback)
		return callback.watcher.all_dicts


class StableBaselinesDDPG(StableBaselinesAgent):
	"""
	This a stable baseline agent using Deep Deterministic Policy Gradient (DDPG) algorithm.
	"""
	name = 'Stable_Baselines_DDPG'

	def _initialize_model(self, marketplace):
		n_actions = marketplace.get_actions_dimension()
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
		self.model = DDPG('MlpPolicy', marketplace, action_noise=action_noise, verbose=False, tensorboard_log=self.tensorboard_log)

	def _load(self, load_path):
		self.model = DDPG.load(load_path, tensorboard_log=self.tensorboard_log)


class StableBaselinesTD3(StableBaselinesAgent):
	"""
	This a stable baseline agent using TD3 which is a direct successor of DDPG.
	"""
	name = 'Stable_Baselines_TD3'

	def _initialize_model(self, marketplace):
		n_actions = marketplace.get_actions_dimension()
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
		self.model = TD3('MlpPolicy', marketplace, action_noise=action_noise, verbose=False, tensorboard_log=self.tensorboard_log)

	def _load(self, load_path):
		self.model = TD3.load(load_path, tensorboard_log=self.tensorboard_log)


class StableBaselinesA2C(StableBaselinesAgent):
	"""
	This a stable baseline agent using A2C.
	"""
	name = 'Stable_Baselines_A2C'

	def _initialize_model(self, marketplace):
		self.model = A2C('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log)

	def _load(self, load_path):
		self.model = A2C.load(load_path, tensorboard_log=self.tensorboard_log)


class StableBaselinesPPO(StableBaselinesAgent):
	"""
	This a stable baseline agent using Proximal Policy Optimization algorithm (PPO).
	"""
	name = 'Stable_Baselines_PPO'

	def _initialize_model(self, marketplace):
		self.model = PPO('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log)

	def _load(self, load_path):
		self.model = PPO.load(load_path, tensorboard_log=self.tensorboard_log)


class StableBaselinesSAC(StableBaselinesAgent):
	"""
	This a stable baseline agent using Soft Actor Critic (SAC).
	"""
	name = 'Stable_Baselines_SAC'

	def _initialize_model(self, marketplace):
		self.model = SAC('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log)

	def _load(self, load_path):
		self.model = SAC.load(load_path, tensorboard_log=self.tensorboard_log)
