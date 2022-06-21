from stable_baselines3 import PPO

from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_rule
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


class StableBaselinesPPO(StableBaselinesAgent):
	"""
	This a stable baseline agent using Proximal Policy Optimization algorithm (PPO).
	"""
	name = 'Stable_Baselines_PPO'

	def _initialize_model(self, marketplace):
		self.model = PPO('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log, **self.config_rl)

	def _load(self, load_path):
		self.model = PPO.load(load_path, tensorboard_log=self.tensorboard_log)

	@staticmethod
	def get_configurable_fields() -> list:
		return [
			('learning_rate', float, between_zero_one_rule),
			('n_steps', int, greater_zero_rule),
			('batch_size', int, greater_zero_rule),
			('n_epochs', int, greater_zero_rule),
			('gamma', float, between_zero_one_rule),
			('clip_range', float, between_zero_one_rule)
		]
