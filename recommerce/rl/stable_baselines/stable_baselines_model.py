import numpy as np
import stable_baselines3.common.monitor
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.market.sim_market import SimMarket
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent
from recommerce.rl.stable_baselines.stable_baselines_callback import PerStepCheck


class StableBaselinesAgent(ReinforcementLearningAgent, LinearAgent, CircularAgent):
	def __init__(self, marketplace=None, optim=None, load_path=None, name=None):
		assert marketplace is not None
		assert isinstance(marketplace, SimMarket), \
			f'if marketplace is provided, marketplace must be a SimMarket, but is {type(marketplace)}'
		assert optim is not None
		assert isinstance(load_path, str)
		assert name is None or isinstance(name, str)

		self.marketplace = marketplace
		if load_path is None:
			self._initialize_model(marketplace)
			print(f'I initiate {self.name}-agent using {self.model.device} device')
		if load_path is not None:
			self._load(load_path)
			print(f'I load {self.name}-agent using {self.model.device} device from {load_path}')

		if name is not None:
			self.name = name

	def policy(self, observation: np.array) -> np.array:
		assert isinstance(observation, np.ndarray), f'{observation}: this is a {type(observation)}, not a np ndarray'
		return self.model.predict(observation)[0]

	def synchronize_tgt_net(self):  # pragma: no cover
		assert False, 'This method may never be used in a StableBaselinesAgent!'

	def train_agent(self, training_steps=100000, iteration_length=500):
		callback = PerStepCheck(type(self), type(self.marketplace), training_steps=training_steps, iteration_length=iteration_length)
		self.model.set_env(stable_baselines3.common.monitor.Monitor(self.marketplace, callback.save_path))
		self.model.learn(training_steps, callback=callback)


class StableBaselinesDDPG(StableBaselinesAgent):
	name = 'Stable_Baselines_DDPG'

	def _initialize_model(self, marketplace):
		n_actions = marketplace.get_actions_dimension()
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
		self.model = DDPG('MlpPolicy', marketplace, action_noise=action_noise, verbose=False)

	def _load(self, load_path):
		self.model = DDPG.load(load_path)


class StableBaselinesTD3(StableBaselinesAgent):
	name = 'Stable_Baselines_TD3'

	def _initialize_model(self, marketplace):
		n_actions = marketplace.get_actions_dimension()
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
		self.model = TD3('MlpPolicy', marketplace, action_noise=action_noise, verbose=False)

	def _load(self, load_path):
		self.model = TD3.load(load_path)


class StableBaselinesA2C(StableBaselinesAgent):
	name = 'Stable_Baselines_A2C'

	def _initialize_model(self, marketplace):
		self.model = A2C('MlpPolicy', marketplace, verbose=False)

	def _load(self, load_path):
		self.model = A2C.load(load_path)


class StableBaselinesPPO(StableBaselinesAgent):
	name = 'Stable_Baselines_PPO'

	def _initialize_model(self, marketplace):
		self.model = PPO('MlpPolicy', marketplace, verbose=False)

	def _load(self, load_path):
		self.model = PPO.load(load_path)


class StableBaselinesSAC(StableBaselinesAgent):
	name = 'Stable_Baselines_SAC'

	def _initialize_model(self, marketplace):
		self.model = SAC('MlpPolicy', marketplace, verbose=False)

	def _load(self, load_path):
		self.model = SAC.load(load_path)
