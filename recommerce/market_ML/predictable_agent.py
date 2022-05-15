from recommerce.configuration.hyperparameter_config import HyperparameterConfig
from recommerce.market.circular.circular_vendors import RuleBasedAgent
from recommerce.market.vendors import Agent


class PredictableAgent(RuleBasedAgent):
	def __init__(self, config: HyperparameterConfig, name='Predicatable Agent'):
		super(Agent, self).__init__(config, name)
		self.step_counter = 0

	def policy(self, observation, *_):
		self.step_counter += 1
		return self.step_counter % self.config.max_price
