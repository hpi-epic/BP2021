from stable_baselines3 import PPO

from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_rule
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


class StableBaselinesPPO(StableBaselinesAgent):
	"""
	This a stable baseline agent using Proximal Policy Optimization algorithm (PPO).
	"""
	name = 'Stable_Baselines_PPO'

	def _initialize_model(self, marketplace):
		self.model = PPO('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log)

	def _load(self, load_path):
		self.model = PPO.load(load_path, tensorboard_log=self.tensorboard_log)

	@staticmethod
	def get_configurable_fields() -> list:
		return [
			('testvalue1', float, between_zero_one_rule),
			('ppovalue', float, greater_zero_rule)
		]
