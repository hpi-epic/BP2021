import os
from abc import ABC, abstractmethod

import numpy as np
from attrdict import AttrDict
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

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
			print(f'Initializing {self.name}-agent using {self.model.device} device')
		if load_path is not None:
			self._load(load_path)
			print(f'Loading {self.name}-agent using {self.model.device} device from {load_path}')

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

	def train_agent(self, training_steps=100001, iteration_length=500, analyze_after_training=True):
		callback = RecommerceCallback(
			type(self), self.marketplace, self.config_market, self.config_rl, training_steps=training_steps,
			iteration_length=iteration_length, signature=self.name, analyze_after_training=analyze_after_training)
		self.model.learn(training_steps, callback=callback)
		self.marketplace.customers_dataframe.to_excel(os.path.join(PathManager.results_path, f'customers_dataframe_{self.name}.xlsx'))
		self.marketplace.owners_dataframe.to_excel(os.path.join(PathManager.results_path, f'owners_dataframe_{self.name}.xlsx'))
		self.marketplace.competitor_reaction_dataframe.to_excel(
			os.path.join(PathManager.results_path, f'competitor_reaction_dataframe_{self.name}.xlsx'))
		return callback.watcher

	def train_with_default_eval(self, training_steps=100001):
		save_path = os.path.join(PathManager.results_path, 'best_model', f'{self.name}')
		log_path = os.path.join(PathManager.results_path, 'logs', f'{self.name}')
		os.makedirs(log_path, exist_ok=True)
		callback = EvalCallback(Monitor(self.marketplace, filename=log_path),
			best_model_save_path=save_path, log_path=log_path, render=False)
		self.model.learn(training_steps, callback=callback)
		self.marketplace.customers_dataframe.to_excel(os.path.join(PathManager.results_path, f'customers_dataframe_{self.name}.xlsx'))
		self.marketplace.owners_dataframe.to_excel(os.path.join(PathManager.results_path, f'owners_dataframe_{self.name}.xlsx'))
		self.marketplace.competitor_reaction_dataframe.to_excel(
			os.path.join(PathManager.results_path, f'competitor_reaction_dataframe_{self.name}.xlsx'))
		return save_path

	@staticmethod
	def get_configurable_fields() -> list:
		raise NotImplementedError
