import matplotlib.pyplot as plt
from recommerce.market.linear.linear_vendors import CompetitorLinearRatio1
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningCERebuyAgent
from recommerce.rl.actorcritic.actorcritic_agent import ContinuosActorCriticAgentFixedOneStd
import os
import recommerce.configuration.path_manager as path_manager
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

    def access_and_adjust_policy(self, policy_value, policy_access):
        policy_value = policy_value if policy_access is None else policy_value[policy_access]
        assert isinstance(policy_value, int)
        return policy_value + 1

    def analyze_policy(self, base_input, analyzed_features, title="<<add a title here>>", policyaccess=None):
        assert isinstance(base_input, np.ndarray), "base_input must be a numpy ndarray"
        assert isinstance(analyzed_features, list), "analyzed_features must be a list containing triples"
        assert 1 <= len(analyzed_features) and len(analyzed_features) <= 2, "you can analyze either one or two features at once"
        assert isinstance(title, str), "title must be a string"
        assert policyaccess is None or isinstance(policyaccess, int), "if policyaccess is not None, policyaccess must be an integer"
        self.assert_analyzed_feature_valid(analyzed_features[0])
        if len(analyzed_features) == 2:
            self.assert_analyzed_feature_valid(analyzed_features[1])
            assert analyzed_features[0][0] != analyzed_features[1][0], "the two entries must analyze different features"
        
        fig, ax = plt.subplots()
        ax.set_xlabel(analyzed_features[0][1])
        if len(analyzed_features) == 1:
            pointsx = []
            pointsy = []
            for x in analyzed_features[0][2]:
                base_input[analyzed_features[0][0]] = x
                y = self.agent_to_analyze.policy(base_input)
                y = self.access_and_adjust_policy(y, policyaccess)
                pointsx.append(x)
                pointsy.append(y)

            plt.scatter(pointsx, pointsy)
            ax.set_ylabel(title)
        else:
            policyval = [[0 for _ in analyzed_features[0][2]] for _ in analyzed_features[1][2]]
            x1base = list(analyzed_features[0][2])[0]
            x2base = list(analyzed_features[1][2])[0]
            for x1 in analyzed_features[0][2]:
                for x2 in analyzed_features[1][2]:
                    base_input[analyzed_features[0][0]] = x1
                    base_input[analyzed_features[1][0]] = x2
                    y = self.agent_to_analyze.policy(base_input)
                    y = self.access_and_adjust_policy(y, policyaccess)
                    policyval[x2 - x2base][x1 - x1base] = y

            myextend = [min(analyzed_features[0][2]), max(analyzed_features[0][2]), min(analyzed_features[1][2]), max(analyzed_features[1][2])]
            p = ax.imshow(policyval, aspect='auto', origin='lower', extent=myextend)
            plt.colorbar(p)
            ax.set_ylabel(analyzed_features[1][1])

        plt.title(title)
        plt.show()

# Analyze the rule based linear competitor
# PolicyAnalyzer(CompetitorLinearRatio1()).analyze_policy(np.array([15, -1, 10]), [(1, "competitor price", range(1, 11))])
# PolicyAnalyzer(CompetitorLinearRatio1()).analyze_policy(np.array([-1, -1, 10]), [(0, "own quality", range(5, 20)), (1, "competitor price", range(1, 11))], "agent's policy")

# Analyze the rule based circular competitor
# PolicyAnalyzer(RuleBasedCERebuyAgent()).analyze_policy(np.array([50, -1]), [(1, "self storage stock", range(0, 30))], "refurbished price", 0)
# PolicyAnalyzer(RuleBasedCERebuyAgent()).analyze_policy(np.array([-1, -1]), [(1, "self storage stock", range(0, 30)), (0, "in circulation", range(0, 200))], "refurbished price", 0)

# Analyze the rule circular monopoly q_learning agent
# q_learing_agent = QLearningCERebuyAgent(2, 1000, load_path=os.path.join(path_manager.PathManager.user_path, 'data', 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'))
# PolicyAnalyzer(q_learing_agent).analyze_policy(np.array([-1, -1]), [(1, "self storage stock", range(0, 30)), (0, "in circulation", range(0, 200))], "refurbished price", 0)
# PolicyAnalyzer(q_learing_agent).analyze_policy(np.array([-1, -1]), [(1, "self storage stock", range(0, 30)), (0, "in circulation", range(0, 200))], "new price", 1)
# PolicyAnalyzer(q_learing_agent).analyze_policy(np.array([-1, -1]), [(1, "self storage stock", range(0, 30)), (0, "in circulation", range(0, 200))], "rebuy price", 2)

# Analyze the rule circular one competitor q_learning agent
# q_learing_agent = ContinuosActorCriticAgentFixedOneStd(6, 3, load_path=os.path.join(path_manager.PathManager.user_path, 'data', 'actor_parametersCircularEconomyRebuyPriceOneCompetitor_ContinuosActorCriticAgentFixedOneStd.dat'))
# PolicyAnalyzer(q_learing_agent).analyze_policy(np.array([75, 10, -1, -1, 2, 12]), [(1, "competitor's refurbished price", range(0, 10)), (0, "competitor's new price", range(0, 10))], "own refurbished price", 0)
# PolicyAnalyzer(q_learing_agent).analyze_policy(np.array([75, 10, -1, -1, 2, 12]), [(1, "competitor's refurbished price", range(0, 10)), (0, "competitor's new price", range(0, 10))], "own new price", 1)
# PolicyAnalyzer(q_learing_agent).analyze_policy(np.array([75, 10, -1, -1, 2, 12]), [(1, "competitor's refurbished price", range(0, 10)), (0, "competitor's new price", range(0, 10))], "own rebuy price", 2)