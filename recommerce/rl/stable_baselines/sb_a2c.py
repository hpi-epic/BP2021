from stable_baselines3 import A2C

from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_rule
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


class StableBaselinesA2C(StableBaselinesAgent):
	"""
	This a stable baseline agent using A2C.
	"""
	name = 'Stable_Baselines_A2C'

	def _initialize_model(self, marketplace):
		self.model = A2C('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log, **self.config_rl)

	def _load(self, load_path):
		self.model = A2C.load(load_path, tensorboard_log=self.tensorboard_log)

	@staticmethod
	def get_configurable_fields() -> list:
		return [
			('learning_rate', float, between_zero_one_rule),
			('n_steps', int, greater_zero_rule),
			('gamma', float, between_zero_one_rule)
		]
