import os

import matplotlib.pyplot as plt
import numpy as np

from recommerce.configuration.path_manager import PathManager


class PolicyAnalyzer():
    def __init__(self, agent_to_analyze):
        self.agent_to_analyze = agent_to_analyze
        self.folder_path = os.path.abspath(os.path.join(PathManager.results_path, 'policyanalyzer'))

    def _assert_analyzed_feature_valid(self, feature):
        assert isinstance(feature, tuple), f'{feature}: such a feature must be a tuple'
        assert len(feature) == 3, f'{feature}: such a feature must be a triple'
        assert isinstance(feature[0], int), f'{feature}: the first entry must be the input index to analyze'
        assert isinstance(feature[1], str), f'{feature}: the second entry must be the description, a string'
        assert isinstance(feature[2], range), f'{feature}: the third entry must be the search range'

    def _access_and_adjust_policy(self, policy_value, policy_access):
        policy_value = policy_value if policy_access is None else policy_value[policy_access]
        assert isinstance(policy_value, int)
        return policy_value + 1

    def analyze_policy(self, base_input, analyzed_features, title='add a title here', policyaccess=None):
        assert isinstance(base_input, np.ndarray), 'base_input must be a numpy ndarray'
        assert isinstance(analyzed_features, list), 'analyzed_features must be a list containing triples'
        assert 1 <= len(analyzed_features) and len(analyzed_features) <= 2, 'you can analyze either one or two features at once'
        assert isinstance(title, str), 'title must be a string'
        assert policyaccess is None or isinstance(policyaccess, int), 'if policyaccess is not None, policyaccess must be an integer'
        self._assert_analyzed_feature_valid(analyzed_features[0])
        if len(analyzed_features) == 2:
            self._assert_analyzed_feature_valid(analyzed_features[1])
            assert analyzed_features[0][0] != analyzed_features[1][0], 'the two entries must analyze different features'

        plt.clf()
        plt.xlabel(analyzed_features[0][1])
        if len(analyzed_features) == 1:
            pointsx = []
            pointsy = []
            for x in analyzed_features[0][2]:
                base_input[analyzed_features[0][0]] = x
                y = self.agent_to_analyze.policy(base_input)
                y = self._access_and_adjust_policy(y, policyaccess)
                pointsx.append(x)
                pointsy.append(y)

            plt.scatter(pointsx, pointsy)
            plt.ylabel(title)
            plt.grid(True)
        else:
            policyval = [[0 for _ in analyzed_features[0][2]] for _ in analyzed_features[1][2]]
            x1base = list(analyzed_features[0][2])[0]
            x2base = list(analyzed_features[1][2])[0]
            for x1 in analyzed_features[0][2]:
                for x2 in analyzed_features[1][2]:
                    base_input[analyzed_features[0][0]] = x1
                    base_input[analyzed_features[1][0]] = x2
                    y = self.agent_to_analyze.policy(base_input)
                    y = self._access_and_adjust_policy(y, policyaccess)
                    policyval[x2 - x2base][x1 - x1base] = y

            myextend = [min(analyzed_features[0][2]), max(analyzed_features[0][2]),
                        min(analyzed_features[1][2]), max(analyzed_features[1][2])]
            p = plt.imshow(policyval, aspect='auto', origin='lower', extent=myextend)
            plt.colorbar(p)
            plt.ylabel(analyzed_features[1][1])

        plt.title(title)
        underscore_title = title.replace(' ', '_')
        plt.savefig(fname=os.path.join(self.folder_path, f'{underscore_title}.png'))
