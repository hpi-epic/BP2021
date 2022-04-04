import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent


class StableBaselinesAgent(ReinforcementLearningAgent, LinearAgent, CircularAgent):
	def __init__(self, marketplace=None, optim=None, load_path=None, name='stable_baselines'):
		assert marketplace is not None
		assert isinstance(marketplace, gym.Env), \
			f'if marketplace is provided, marketplace must be a SimMarket, but is {type(marketplace)}'
		self._initialize_model(marketplace)
		self.name = name
		if load_path is not None:
			self.model.load(load_path)

	def policy(self, observation):
		return self.model.predict(observation)[0]

	def synchronize_tgt_net(self):  # pragma: no cover
		assert False, 'This method may never be used in a StableBaselinesAgent!'


class StableBaselinesDDPG(StableBaselinesAgent):
	def _initialize_model(self, marketplace):
		n_actions = marketplace.get_actions_dimension()
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
		self.model = DDPG('MlpPolicy', marketplace, action_noise=action_noise, verbose=False)
		print(f'I initiate an Stable-Baselines-DDPG-agent using {self.model.device} device')
