from unittest.mock import patch

import pytest
from attrdict import AttrDict

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.monitoring.agent_monitoring.am_monitoring as monitoring
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.hyperparameter_config import HyperparameterConfigLoader
from recommerce.market.circular.circular_vendors import FixedPriceCEAgent, FixedPriceCERebuyAgent, HumanPlayerCERebuy, RuleBasedCEAgent
from recommerce.market.linear.linear_vendors import FixedPriceLEAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent

monitor = monitoring.Monitor()

config_market: AttrDict = HyperparameterConfigLoader.load('market_config', circular_market.CircularEconomyRebuyPriceMonopoly)


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(
		separate_markets=False,
		episodes=50,
		plot_interval=10,
		marketplace=circular_market.CircularEconomyMonopoly,
		agents=[(FixedPriceCERebuyAgent, [])],
		config_market=config_market)


def test_get_modelfile_path():
	with patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as mock_exists, \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'):
		mock_exists.return_value = False
		with pytest.raises(AssertionError) as assertion_message:
			monitor.configurator._get_modelfile_path('non_existing_modelfile')
		assert 'Modelfiles must end in .dat or .zip: non_existing_modelfile' in str(assertion_message.value)


incorrect_update_agents_RL_testcases = [
	([(QLearningAgent, ['modelfile.dat', 'arg', 'too_much'])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(QLearningAgent, [1, 2, 3, 4])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(QLearningAgent, ['modelfile.dat', 35])], 'the arguments for a RL-agent must be of type str'),
	([(QLearningAgent, [25])], 'the arguments for a RL-agent must be of type str'),
	([(QLearningAgent, ['agent_name', 'modelfile.dat'])], 'if two arguments as well as a config are provided, ' +
		'the first extra one must be the modelfile.'),
	([(QLearningAgent, ['mymodel.dat'])], 'the specified modelfile does not exist')
]


@pytest.mark.parametrize('agents, expected_message', incorrect_update_agents_RL_testcases)
def test_incorrect_update_agents_RL(agents, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		monitor.configurator.setup_monitoring(agents=agents, config_market=config_market)
	assert expected_message in str(assertion_message.value)


correct_update_agents_RL_testcases = [
	[(QLearningAgent, [])],
	[(QLearningAgent, ['new_name'])],
	[(QLearningAgent, ['CircularEconomyMonopoly_QLearningAgent.dat'])],
	[(QLearningAgent, ['CircularEconomyMonopoly_QLearningAgent.dat', 'new_name'])],
	[(QLearningAgent, [f'{circular_market.CircularEconomyMonopoly.__name__}_{QLearningAgent.__name__}.dat'])]
]


@pytest.mark.parametrize('agents', correct_update_agents_RL_testcases)
def test_correct_update_agents_RL(agents):
	monitor.configurator.setup_monitoring(agents=agents, config_market=config_market)


def test_correct_setup_monitoring():
	monitor.configurator.setup_monitoring(
		separate_markets=False,
		episodes=10,
		plot_interval=2,
		marketplace=circular_market.CircularEconomyRebuyPriceDuopoly,
		agents=[(HumanPlayerCERebuy, ['reptiloid']),
			(QLearningAgent, ['CircularEconomyRebuyPriceDuopoly_QLearningAgent.dat', 'q_learner'])],
		config_market=config_market)
	assert monitor.configurator.separate_markets is False
	assert 10 == monitor.configurator.episodes
	assert 2 == monitor.configurator.plot_interval
	assert isinstance(monitor.configurator.marketplace, circular_market.CircularEconomyRebuyPriceDuopoly)
	assert 2 == len(monitor.configurator.agents)
	assert isinstance(monitor.configurator.agents[0], HumanPlayerCERebuy)
	assert isinstance(monitor.configurator.agents[1], QLearningAgent)
	assert 'reptiloid' == monitor.configurator.agents[0].name
	assert 'q_learner' == monitor.configurator.agents[1].name
	assert 2 == len(monitor.configurator.agent_colors)


setting_multiple_agents_testcases = [
	[(FixedPriceCERebuyAgent, []), (FixedPriceCEAgent, [])],
	[(FixedPriceCERebuyAgent, []), (FixedPriceCEAgent, []), (FixedPriceCERebuyAgent, [])]
]


@pytest.mark.parametrize('agents', setting_multiple_agents_testcases)
def test_setting_multiple_agents(agents):
	monitor.configurator.setup_monitoring(separate_markets=True, agents=agents, config_market=config_market)


def test_setting_market_not_agents():
	monitor.configurator.setup_monitoring(marketplace=circular_market.CircularEconomyMonopoly, config_market=config_market)


correct_setup_monitoring_testcases = [
	({'marketplace': linear_market.LinearEconomyDuopoly,
		'agents': [(QLearningAgent, ['LinearEconomyDuopoly_QLearningAgent.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopoly,
		'agents': [(QLearningAgent,
		['CircularEconomyRebuyPriceMonopoly_QLearningAgent.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopoly,
		'agents': [(actorcritic_agent.ContinuousActorCriticAgentEstimatingStd,
		['actor_parametersCircularEconomyRebuyPriceMonopoly_ContinuousActorCriticAgentEstimatingStd.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceDuopoly,
		'agents': [(actorcritic_agent.ContinuousActorCriticAgentFixedOneStd,
		['actor_parametersCircularEconomyRebuyPriceDuopoly_ContinuousActorCriticAgentFixedOneStd.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceDuopoly,
		'agents': [(actorcritic_agent.DiscreteActorCriticAgent,
		['actor_parametersCircularEconomyRebuyPriceDuopoly_DiscreteACACircularEconomyRebuy.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceDuopoly,
		'agents': [(QLearningAgent,
		['CircularEconomyRebuyPriceDuopoly_QLearningAgent.dat'])]})
]


@pytest.mark.parametrize('parameters', correct_setup_monitoring_testcases)
def test_correct_setup_monitoring_parametrized(parameters):
	dict = {
		'separate_markets': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	monitor.configurator.setup_monitoring(
		separate_markets=dict['separate_markets'],
		episodes=dict['episodes'],
		plot_interval=dict['plot_interval'],
		marketplace=dict['marketplace'],
		agents=dict['agents'],
		config_market=config_market
	)


incorrect_setup_monitoring_testcases = [
	({'separate_markets': 1}, 'separate_markets must be a Boolean'),
	({'episodes': 'Hello World'}, 'episodes must be of type int'),
	({'episodes': 0}, 'episodes must not be 0'),
	({'plot_interval': '1'}, 'plot_interval must be of type int'),
	({'plot_interval': 0}, 'plot_interval must not be 0'),
	({'episodes': 4, 'plot_interval': 5}, 'plot_interval must be <= episodes, or no plots can be generated.'),
	({'marketplace': RuleBasedCEAgent}, 'the marketplace must be a subclass of SimMarket'),
	({'marketplace': linear_market.LinearEconomyDuopoly, 'agents': [(RuleBasedCEAgent, [])]},
		'If the market is linear, the agent must be linear too!'),
	({'agents': [RuleBasedCEAgent]}, 'agents must be a list of tuples'),
	({'agents': [[RuleBasedCEAgent, 1, '2']]}, 'agents must be a list of tuples'),
	({'agents': [(RuleBasedCEAgent)]}, 'agents must be a list of tuples'),
	({'agents': [FixedPriceLEAgent, FixedPriceCERebuyAgent]}, 'agents must be a list of tuples'),
	({'agents': [(RuleBasedCEAgent, ['arg'], 'too_much')]},
		'the list entries in agents must have size 2 ([agent_class, arguments])'),
	({'agents': [(linear_market.LinearEconomyDuopoly, [])]}, 'the first entry in each agent-tuple must be an agent class in `vendors.py`'),
	({'agents': [(RuleBasedCEAgent, linear_market.LinearEconomyDuopoly)]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(RuleBasedCEAgent, 'new_name')]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(RuleBasedCEAgent, []), (FixedPriceLEAgent, [])]},
		'the agents must all be of the same type (Linear/Circular)'),
	({'agents': [(RuleBasedCEAgent, []), (FixedPriceLEAgent, []), (FixedPriceCEAgent, [])]},
		'the agents must all be of the same type (Linear/Circular)'),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopoly, 'agents': [(FixedPriceLEAgent, [])]},
		'If the market is circular, the agent must be circular too!'),
	({'marketplace': linear_market.LinearEconomyDuopoly, 'agents': [(FixedPriceCEAgent, [])]},
		'If the market is linear, the agent must be linear too!'),
	({'marketplace': linear_market.LinearEconomyOligopoly,
		'agents': [(QLearningAgent,
		['CircularEconomyRebuyPriceMonopoly_QLearningAgent.dat'])]},
		'The modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopoly,
		'agents': [(actorcritic_agent.ContinuousActorCriticAgentFixedOneStd,
		['actor_parametersCircularEconomyRebuyPriceMonopoly_ContinuousActorCriticAgentEstimatingStd.dat'])]},
		'The modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyRebuyPriceDuopoly,
		'agents': [(actorcritic_agent.DiscreteActorCriticAgent,
		['actor_parametersCircularEconomyRebuyPriceDuopoly_ContinuousActorCriticAgentFixedOneStd.dat'])]},
		'The modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyMonopoly,
		'agents': [(actorcritic_agent.DiscreteActorCriticAgent,
		['actor_parametersCircularEconomyRebuyPriceDuopoly_DiscreteACACircularEconomyRebuy.dat'])]},
		'The modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyMonopoly,
		'agents': [(QLearningAgent,
		['CircularEconomyRebuyPriceDuopoly_QLearningAgent.dat'])]},
		'The modelfile is not compatible with the agent you tried to instantiate')
]


@pytest.mark.parametrize('parameters, expected_message', incorrect_setup_monitoring_testcases)
def test_incorrect_setup_monitoring(parameters, expected_message):
	dict = {
		'separate_markets': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	with pytest.raises(Exception) as assertion_message:
		monitor.configurator.setup_monitoring(
			separate_markets=dict['separate_markets'],
			episodes=dict['episodes'],
			plot_interval=dict['plot_interval'],
			marketplace=dict['marketplace'],
			agents=dict['agents'],
			config_market=config_market
		)
	assert expected_message in str(assertion_message.value)


incorrect_setup_monitoring_type_errors_testcases = [
	{'marketplace': linear_market.LinearEconomyDuopoly(config=config_market)},
	{'agents': [(linear_market.LinearEconomyDuopoly(config=config_market), [])]},
	{'agents': [(RuleBasedCEAgent(config_market=config_market), [])]}
]


@pytest.mark.parametrize('parameters', incorrect_setup_monitoring_type_errors_testcases)
def test_incorrect_setup_monitoring_type_errors(parameters):
	dict = {
		'separate_markets': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	with pytest.raises(TypeError):
		monitor.configurator.setup_monitoring(
			separate_markets=dict['separate_markets'],
			episodes=dict['episodes'],
			plot_interval=dict['plot_interval'],
			marketplace=dict['marketplace'],
			agents=dict['agents'],
			config_market=config_market
		)


print_configuration_testcases = [
	([(RuleBasedCEAgent, [])]),
	([(RuleBasedCEAgent, []), (FixedPriceCEAgent, [])])
]


@pytest.mark.parametrize('agents', print_configuration_testcases)
def test_print_configuration(agents):
	monitor.configurator.setup_monitoring(separate_markets=True, agents=agents, config_market=config_market)

	monitor.configurator.print_configuration()


@pytest.mark.parametrize('agents', print_configuration_testcases)
def test_print_configuration_ratio(agents):
	monitor.configurator.setup_monitoring(separate_markets=True, config_market=config_market, episodes=51, plot_interval=1, agents=agents)

	with patch('recommerce.monitoring.agent_monitoring.am_configuration.input', create=True) as mocked_input:
		mocked_input.side_effect = ['n']
		monitor.configurator.print_configuration()
