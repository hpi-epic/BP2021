from stable_baselines3 import SAC

from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_rule
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


class StableBaselinesSAC(StableBaselinesAgent):
	"""
	This a stable baseline agent using Soft Actor Critic (SAC).
	"""
	name = 'Stable_Baselines_SAC'

	def _initialize_model(self, marketplace):
		self.model = SAC('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log)

	def _load(self, load_path):
		self.model = SAC.load(load_path, tensorboard_log=self.tensorboard_log)

	@staticmethod
	def get_configurable_fields() -> list:
		return [
			('testvalue1', float, between_zero_one_rule),
			('sacvalue', float, greater_zero_rule)
		]
