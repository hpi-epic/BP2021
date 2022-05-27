import os

import matplotlib.pyplot as plt
import numpy as np

import recommerce.configuration.utils as ut
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgentCompetitive
from recommerce.rl.actorcritic.actorcritic_agent import ContinuousActorCriticAgent


class PolicyAnalyzer():
	def __init__(self, agent_to_analyze, subfolder_name=''):
		self.agent_to_analyze = agent_to_analyze
		self.folder_path = os.path.abspath(os.path.join(PathManager.results_path, 'policyanalyzer', subfolder_name))
		os.makedirs(self.folder_path, exist_ok=True)
		ut.ensure_results_folders_exist()

	def _agents_policy(self, observation) -> tuple:
		"""
		Some agents generate a randomized policy.
		This method is made to wrap the policy call.
		If necessary, it will ask for the mean parameter and not a shuffled value.

		Args:
			observation (np.array): The observation which should be used.

		Returns:
			tuple: The requested policy value
		"""
		if isinstance(self.agent_to_analyze, ContinuousActorCriticAgent):
			return self.agent_to_analyze.policy(observation, mean_only=True)
		else:
			return self.agent_to_analyze.policy(observation)

	def _assert_analyzed_feature_valid(self, feature):
		assert isinstance(feature, tuple), f'{feature}: such a feature must be a tuple'
		assert len(feature) == 3, f'{feature}: such a feature must be a triple'
		assert isinstance(feature[0], int), f'{feature}: the first entry must be the input index to analyze'
		assert isinstance(feature[1], str), f'{feature}: the second entry must be the description, a string'
		assert isinstance(feature[2], range), f'{feature}: the third entry must be the search range'

	def _access_and_adjust_policy(self, policy_value, policy_access):
		policy_value = policy_value if policy_access is None else policy_value[policy_access]
		assert isinstance(policy_value, (int, float)), f'policy_value must be an int or float but is {policy_value}, a {type(policy_value)}'
		return policy_value + 1

	def _plot_one_depending_variable(self, base_input, analyzed_features, title, policyaccess):
		pointsx = []
		pointsy = []
		for x in analyzed_features[0][2]:
			base_input[analyzed_features[0][0]] = x
			y = self._agents_policy(base_input)
			y = self._access_and_adjust_policy(y, policyaccess)
			pointsx.append(x)
			pointsy.append(y)

		plt.scatter(pointsx, pointsy)
		plt.ylabel(title)
		plt.grid(True)

	def _plot_two_depending_variables(self, base_input, analyzed_features, title, policyaccess):
		policyval = [[0 for _ in analyzed_features[0][2]] for _ in analyzed_features[1][2]]
		x1base = list(analyzed_features[0][2])[0]
		x2base = list(analyzed_features[1][2])[0]
		for x1 in analyzed_features[0][2]:
			for x2 in analyzed_features[1][2]:
				base_input[analyzed_features[0][0]] = x1
				base_input[analyzed_features[1][0]] = x2
				y = self._agents_policy(base_input)
				y = self._access_and_adjust_policy(y, policyaccess)
				policyval[x2 - x2base][x1 - x1base] = y

		shown_section = [min(analyzed_features[0][2]), max(analyzed_features[0][2]),
					min(analyzed_features[1][2]), max(analyzed_features[1][2])]
		p = plt.imshow(policyval, aspect='auto', origin='lower', extent=shown_section)
		plt.colorbar(p)
		plt.ylabel(analyzed_features[1][1])

	def analyze_policy(self, base_input, analyzed_features, title='add a title here', policyaccess=None) -> str:
		"""
		This method generates a pyplotlib diagram which visualizes the policy.
		Because an observation can be high-dimensional, base_input gives a template in which up to two combinations of features can be inserted.

		Args:
			base_input (np.array): The template for an observation accepted by the agent
			analyzed_features (list): The list of one or two features to analyze. Look the assert for further details
			title (str, optional): You can provide a title to your graphics.
			The diagram will be saved under this title. Defaults to 'add a title here'.
			policyaccess (int, optional): If the policy-output is more-dimensional, policyaccess says which output to take. Defaults to None.

		Returns:
			str: The path to the saved diagram.
		"""
		assert isinstance(base_input, np.ndarray), 'base_input must be a numpy ndarray'
		assert isinstance(analyzed_features, list), 'analyzed_features must be a list containing triples'
		assert 1 <= len(analyzed_features) <= 2, 'you can analyze either one or two features at once'
		assert isinstance(title, str), 'title must be a string'
		assert policyaccess is None or isinstance(policyaccess, int), 'if policyaccess is not None, policyaccess must be an integer'
		self._assert_analyzed_feature_valid(analyzed_features[0])
		if len(analyzed_features) == 2:
			self._assert_analyzed_feature_valid(analyzed_features[1])
			assert analyzed_features[0][0] != analyzed_features[1][0], 'the two entries must analyze different features'

		plt.clf()
		plt.xlabel(analyzed_features[0][1])
		if len(analyzed_features) == 1:
			self._plot_one_depending_variable(base_input, analyzed_features, title, policyaccess)
		else:
			self._plot_two_depending_variables(base_input, analyzed_features, title, policyaccess)

		plt.title(title)
		underscore_title = title.replace(' ', '_')
		savepath = os.path.join(self.folder_path, f'{underscore_title}.png')
		plt.savefig(fname=savepath)
		return savepath


if __name__ == '__main__':
	config_market = HyperparameterConfigLoader.load('market_config')
	pa = PolicyAnalyzer(RuleBasedCERebuyAgentCompetitive(config_market=config_market), 'default_configuration')
	one_competitor_examples = [
		('rule based own refurbished price', 0),
		('rule based own new price', 1),
		('rule based own rebuy price', 2)
	]
	for example in one_competitor_examples:
		pa.analyze_policy(
			np.array([75, 10, -1, -1, 2, 12]),
			[(2, "competitor's refurbished price", range(10)), (3, "competitor's new price", range(10))],
			*example
		)
