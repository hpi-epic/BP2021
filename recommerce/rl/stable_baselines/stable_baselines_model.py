import gym
import numpy as np
from stable_baselines3 import DDPG, TD3, A2C, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
import stable_baselines3.common.monitor

from recommerce.market.circular.circular_vendors import CircularAgent
from recommerce.market.linear.linear_vendors import LinearAgent
from recommerce.rl.reinforcement_learning_agent import ReinforcementLearningAgent
from recommerce.rl.stable_baselines.stable_baselines_training import PerStepCheck
import recommerce.market.circular.circular_sim_market as circular_market


class StableBaselinesAgent(ReinforcementLearningAgent, LinearAgent, CircularAgent):
	def __init__(self, marketplace=None, optim=None, load_path=None, iteration_length=500, name='enter a name here'):
		assert marketplace is not None
		assert isinstance(marketplace, gym.Env), \
			f'if marketplace is provided, marketplace must be a SimMarket, but is {type(marketplace)}'
		self.callback = PerStepCheck(type(self), type(marketplace), iteration_length=iteration_length)
		marketplace = stable_baselines3.common.monitor.Monitor(marketplace, self.callback.save_path)
		self._initialize_model(marketplace)
		if load_path is not None:
			self.model.load(load_path)
		print(f'I initiate {self.name}-agent using {self.model.device} device')

	def policy(self, observation):
		return self.model.predict(observation)[0]

	def synchronize_tgt_net(self):  # pragma: no cover
		assert False, 'This method may never be used in a StableBaselinesAgent!'

	def train_agent(self, training_steps=100000):
		print(f'Now I start the training with {training_steps} steps')
		self.model.learn(training_steps, callback=self.callback)


class StableBaselinesDDPG(StableBaselinesAgent):
	def _initialize_model(self, marketplace):
		n_actions = marketplace.get_actions_dimension()
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
		self.model = DDPG('MlpPolicy', marketplace, action_noise=action_noise, verbose=False)
		self.name = 'Stable_Baselines_DDPG'


class StableBaselinesTD3(StableBaselinesAgent):
	def _initialize_model(self, marketplace):
		n_actions = marketplace.get_actions_dimension()
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions))
		self.model = TD3('MlpPolicy', marketplace, action_noise=action_noise, verbose=False)
		self.name = 'Stable_Baselines_TD3'


class StableBaselinesA2C(StableBaselinesAgent):
	def _initialize_model(self, marketplace):
		self.model = A2C('MlpPolicy', marketplace, verbose=False)
		self.name = 'Stable_Baselines_A2C'


class StableBaselinesPPO(StableBaselinesAgent):
	def _initialize_model(self, marketplace):
		self.model = PPO('MlpPolicy', marketplace, verbose=False)
		self.name = 'Stable_Baselines_PPO'


class StableBaselinesSAC(StableBaselinesAgent):
	def _initialize_model(self, marketplace):
		self.model = SAC('MlpPolicy', marketplace, verbose=False)
		self.name = 'Stable_Baselines_SAC'


if __name__ == '__main__':
	StableBaselinesPPO(circular_market.CircularEconomyRebuyPriceOneCompetitor(True)).train_agent()
