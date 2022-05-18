

from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.market.circular.circular_vendors import CircularAgent, RuleBasedAgent

# from recommerce.market.vendors import Agent


class PredictableAgent(RuleBasedAgent, CircularAgent):
	def __init__(self, config: HyperparameterConfig, name='Predicatable Agent'):
		super(CircularAgent, self).__init__(config=config, name=name)
		self.step_counter = 0

	def policy(self, observation, *_):
		self.step_counter += 1
		return [(self.step_counter + 1) % self.config.max_price,
			(self.step_counter + 0) % self.config.max_price,
			(self.step_counter + -2) % self.config.max_price]


class PredictableCompetitor(RuleBasedAgent, CircularAgent):
	def __init__(self, config: HyperparameterConfig, name='Predicatable Competitor'):
		super(CircularAgent, self).__init__(config=config, name=name)

	def policy(self, observation, *_):
		result = [observation[2], observation[3], observation[4]]
		print(result)
		return result
