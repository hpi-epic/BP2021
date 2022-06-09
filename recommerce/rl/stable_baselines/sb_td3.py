import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_rule
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


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

	@staticmethod
	def get_configurable_fields() -> list:
		return [
			('testvalue1', float, between_zero_one_rule),
			('td3value', float, greater_zero_rule)
		]
