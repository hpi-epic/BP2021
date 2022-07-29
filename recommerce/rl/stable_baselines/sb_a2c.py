from stable_baselines3 import A2C

from recommerce.configuration.common_rules import between_zero_one_rule, greater_zero_rule
from recommerce.rl.stable_baselines.stable_baselines_model import StableBaselinesAgent


class StableBaselinesA2C(StableBaselinesAgent):
	"""
	This a stable baseline agent using A2C.
	"""
	name = 'Stable_Baselines_A2C'

	def _initialize_model(self, marketplace):
		hidden_neurons = self.config_rl['neurones_per_hidden_layer']
		policy_kwargs = {'net_arch': [dict(pi=[hidden_neurons, hidden_neurons], vf=[hidden_neurons, hidden_neurons])]}
		self.model = A2C('MlpPolicy', marketplace, verbose=False, tensorboard_log=self.tensorboard_log, policy_kwargs=policy_kwargs,
			**{i: self.config_rl[i] for i in self.config_rl if i != 'neurones_per_hidden_layer'})

	def _load(self, load_path):
		self.model = A2C.load(load_path, tensorboard_log=self.tensorboard_log)

	@staticmethod
	def get_configurable_fields() -> list:
		return [
			('learning_rate', float, between_zero_one_rule),
			('n_steps', int, greater_zero_rule),
			('gamma', float, between_zero_one_rule),
			('neurones_per_hidden_layer', int, greater_zero_rule)
		]
