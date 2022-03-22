import matplotlib.pyplot as plt
from recommerce.market.linear.linear_vendors import CompetitorLinearRatio1
from recommerce.rl.q_learning.q_learning_agent import QLearningCEAgent
import numpy as np

class PolicyAnalyzer():
    def __init__(self, agent_to_analyze):
        self.agent_to_analyze = agent_to_analyze

    def assert_analyzed_feature_valid(self, feature):
        assert isinstance(feature, tuple), f"{feature}: such a feature must be a tuple"
        assert len(feature) == 3, f"{feature}: such a feature must be a triple"
        assert isinstance(feature[0], int), f"{feature}: the first entry must be the input index to analyze"
        assert isinstance(feature[1], str), f"{feature}: the second entry must be the description, a string"
        assert isinstance(feature[2], range), f"{feature}: the third entry must be the search range"

    def analyze_policy(self, base_input, analyzed_features):
        assert isinstance(base_input, np.ndarray), "base_input must be a numpy ndarray"
        assert isinstance(analyzed_features, list), "analyzed_features must be a list containing triples"
        assert 1 <= len(analyzed_features) and len(analyzed_features) <= 2, "you can analyze either one or two features at once"
        self.assert_analyzed_feature_valid(analyzed_features[0])
        if len(analyzed_features) == 2:
            self.assert_analyzed_feature_valid(analyzed_features[1])
        
        if len(analyzed_features) == 1:
            pointsx = []
            pointsy = []
            for x in analyzed_features[0][2]:
                base_input[analyzed_features[0][0]] = x
                y = self.agent_to_analyze.policy(base_input)
                pointsx.append(x)
                pointsy.append(y)

            plt.scatter(pointsx, pointsy)
            plt.show()

PolicyAnalyzer(CompetitorLinearRatio1()).analyze_policy(np.array([15, -1, 10]), [(1, "competitor price", range(1, 11))])