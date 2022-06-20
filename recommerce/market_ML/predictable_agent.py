

import random

from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.market.circular.circular_vendors import CircularAgent, RuleBasedAgent

# from recommerce.market.vendors import Agent


class PredictableAgent(RuleBasedAgent, CircularAgent):
	def __init__(self, config: HyperparameterConfig, name='Predicatable Agent'):
		super(CircularAgent, self).__init__(config=config, name=name)
		self.step_counter = 0

	def policy(self, observation, *_):
		self.step_counter = random.randint(0, 1000)
		prices = [round(self.step_counter / 100) % self.config.max_price,
			round(self.step_counter / 10) % self.config.max_price,
			(self.step_counter) % self.config.max_price]
		return prices

		# return [6, 7, 3]


class PredictableCompetitor(RuleBasedAgent, CircularAgent):
	def __init__(self, config: HyperparameterConfig, name='Predicatable Competitor'):
		super(CircularAgent, self).__init__(config=config, name=name)

	def policy(self, observation, *_):
		result = [int(max(0, observation[2] - 1)), int(max(0, observation[3] - 1)), int(max(0, observation[4] - 1))]
		# print(result)
		# result = [5, 8, 4]
		return result
