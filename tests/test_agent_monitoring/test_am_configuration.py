import os
import re
import shutil
from unittest.mock import patch

import pytest

import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
import recommerce.monitoring.agent_monitoring.am_monitoring as monitoring
import recommerce.rl.actorcritic.actorcritic_agent as actorcritic_agent
from recommerce.configuration.path_manager import PathManager
from recommerce.market.circular.circular_vendors import FixedPriceCEAgent, FixedPriceCERebuyAgent, HumanPlayerCERebuy, RuleBasedCEAgent
from recommerce.market.linear.linear_vendors import FixedPriceLEAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningCEAgent, QLearningCERebuyAgent, QLearningLEAgent

monitor = monitoring.Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(
		enable_live_draw=False,
		episodes=50,
		plot_interval=10,
		marketplace=circular_market.CircularEconomyMonopolyScenario(),
		agents=[(FixedPriceCERebuyAgent, [])],
		subfolder_name=f'test_plots_{function.__name__}')


def teardown_module(module):
	for file_name in os.listdir(os.path.join(PathManager.results_path, 'monitoring')):
		if re.match('test_*', file_name):
			assert False, 'Test files were not mocked correctly'


def test_get_folder():
	# if you change the name of this function, change it here as well!
	foldername = 'test_plots_test_get_folder'
	monitor.configurator.get_folder()
	assert os.path.exists(os.path.abspath(os.path.join(PathManager.results_path, 'monitoring', foldername)))
	shutil.rmtree(os.path.join(PathManager.results_path, 'monitoring', foldername))


def test_get_modelfile_path():
	with patch('recommerce.monitoring.agent_monitoring.am_configuration.os.path.exists') as mock_exists, \
		patch('recommerce.monitoring.agent_monitoring.am_configuration.os.makedirs'):
		mock_exists.return_value = False
		with pytest.raises(AssertionError) as assertion_message:
			monitor.configurator._get_modelfile_path('non_existing_modelfile')
		assert 'the specified modelfile does not exist' in str(assertion_message.value)


incorrect_update_agents_RL_testcases = [
	([(QLearningCEAgent, ['modelfile.dat', 'arg', 'too_much'])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(QLearningCEAgent, [1, 2, 3, 4])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(QLearningCEAgent, ['modelfile.dat', 35])], 'the arguments for a RL-agent must be of type str'),
	([(QLearningCEAgent, [25])], 'the arguments for a RL-agent must be of type str'),
	([(QLearningCEAgent, ['agent_name', 'modelfile.dat'])], 'if two arguments are provided, the first one must be the modelfile.'),
	([(QLearningCEAgent, ['mymodel.dat'])], 'the specified modelfile does not exist')
]


@pytest.mark.parametrize('agents, expected_message', incorrect_update_agents_RL_testcases)
def test_incorrect_update_agents_RL(agents, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		monitor.configurator.setup_monitoring(agents=agents)
	assert expected_message in str(assertion_message.value)


correct_update_agents_RL_testcases = [
	[(QLearningCEAgent, [])],
	[(QLearningCEAgent, ['new_name'])],
	[(QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat'])],
	[(QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'new_name'])],
	[(QLearningCEAgent, [f'{circular_market.CircularEconomyMonopolyScenario.__name__}_{QLearningCEAgent.__name__}.dat'])]
]


@pytest.mark.parametrize('agents', correct_update_agents_RL_testcases)
def test_correct_update_agents_RL(agents):
	monitor.configurator.setup_monitoring(agents=agents)


def test_correct_setup_monitoring():
	monitor.configurator.setup_monitoring(
		enable_live_draw=False,
		episodes=10,
		plot_interval=2,
		marketplace=circular_market.CircularEconomyMonopolyScenario(),
		agents=[(HumanPlayerCERebuy, ['reptiloid']),
			(QLearningCERebuyAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'q_learner'])],
		subfolder_name='subfoldername')
	assert monitor.configurator.enable_live_draw is False
	assert 10 == monitor.configurator.episodes
	assert 2 == monitor.configurator.plot_interval
	assert isinstance(monitor.configurator.marketplace, circular_market.CircularEconomyMonopolyScenario)
	assert 2 == len(monitor.configurator.agents)
	assert isinstance(monitor.configurator.agents[0], HumanPlayerCERebuy)
	assert isinstance(monitor.configurator.agents[1], QLearningCERebuyAgent)
	assert 'reptiloid' == monitor.configurator.agents[0].name
	assert 'q_learner' == monitor.configurator.agents[1].name
	assert (os.path.normcase(os.path.abspath(os.path.join(PathManager.results_path, 'monitoring', 'subfoldername')))
		== os.path.normcase(os.path.abspath(monitor.configurator.folder_path)))
	assert 2 == len(monitor.configurator.agent_colors)


setting_multiple_agents_testcases = [
	[(FixedPriceCERebuyAgent, []), (FixedPriceCEAgent, [])],
	[(FixedPriceCERebuyAgent, []), (FixedPriceCEAgent, []), (FixedPriceCERebuyAgent, [])]
]


@pytest.mark.parametrize('agents', setting_multiple_agents_testcases)
def test_setting_multiple_agents(agents):
	monitor.configurator.setup_monitoring(agents=agents)


def test_setting_market_not_agents():
	monitor.configurator.setup_monitoring(marketplace=circular_market.CircularEconomyMonopolyScenario())


correct_setup_monitoring_testcases = [
	({'marketplace': linear_market.ClassicScenario(),
		'agents': [(QLearningLEAgent, ['ClassicScenario_QLearningLEAgent.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopolyScenario(),
		'agents': [(QLearningCERebuyAgent,
		['CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopolyScenario(),
		'agents': [(actorcritic_agent.ContinuosActorCriticAgentEstimatingStd,
		['actor_parametersCircularEconomyRebuyPriceMonopolyScenario_ContinuosActorCriticAgentEstimatingStd.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceOneCompetitor(),
		'agents': [(actorcritic_agent.ContinuosActorCriticAgentFixedOneStd,
		['actor_parametersCircularEconomyRebuyPriceOneCompetitor_ContinuosActorCriticAgentFixedOneStd.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceOneCompetitor(),
		'agents': [(actorcritic_agent.DiscreteACACircularEconomyRebuy,
		['actor_parametersCircularEconomyRebuyPriceOneCompetitor_DiscreteACACircularEconomyRebuy.dat'])]}),
	({'marketplace': circular_market.CircularEconomyRebuyPriceOneCompetitor(),
		'agents': [(QLearningCERebuyAgent,
		['CircularEconomyRebuyPriceOneCompetitor_QLearningCERebuyAgent.dat'])]})
]


@pytest.mark.parametrize('parameters', correct_setup_monitoring_testcases)
def test_correct_setup_monitoring_parametrized(parameters):
	dict = {
		'enable_live_draw': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None,
		'subfolder_name': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	monitor.configurator.setup_monitoring(
		enable_live_draw=dict['enable_live_draw'],
		episodes=dict['episodes'],
		plot_interval=dict['plot_interval'],
		marketplace=dict['marketplace'],
		agents=dict['agents'],
		subfolder_name=dict['subfolder_name']
	)


incorrect_setup_monitoring_testcases = [
	({'enable_live_draw': 1}, 'enable_live_draw must be a Boolean'),
	({'episodes': 'Hello World'}, 'episodes must be of type int'),
	({'episodes': 0}, 'episodes must not be 0'),
	({'plot_interval': '1'}, 'plot_interval must be of type int'),
	({'plot_interval': 0}, 'plot_interval must not be 0'),
	({'episodes': 4, 'plot_interval': 5}, 'plot_interval must be <= episodes, or no plots can be generated.'),
	({'marketplace': RuleBasedCEAgent}, 'the marketplace must be a subclass of SimMarket'),
	({'marketplace': linear_market.ClassicScenario(), 'agents': [(RuleBasedCEAgent, [])]},
		'the agent and marketplace must be of the same economy type'),
	({'agents': [RuleBasedCEAgent]}, 'agents must be a list of tuples'),
	({'agents': [[RuleBasedCEAgent, 1, '2']]}, 'agents must be a list of tuples'),
	({'agents': [(RuleBasedCEAgent)]}, 'agents must be a list of tuples'),
	({'agents': [FixedPriceLEAgent, FixedPriceCERebuyAgent]}, 'agents must be a list of tuples'),
	({'agents': [(RuleBasedCEAgent, ['arg'], 'too_much')]},
		'the list entries in agents must have size 2 ([agent_class, arguments])'),
	({'agents': [(linear_market.ClassicScenario, [])]}, 'the first entry in each agent-tuple must be an agent class in `vendors.py`'),
	({'agents': [(RuleBasedCEAgent, linear_market.ClassicScenario)]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(RuleBasedCEAgent, 'new_name')]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(RuleBasedCEAgent, []), (FixedPriceLEAgent, [])]},
		'the agents must all be of the same type (Linear/Circular)'),
	({'agents': [(RuleBasedCEAgent, []), (FixedPriceLEAgent, []), (FixedPriceCEAgent, [])]},
		'the agents must all be of the same type (Linear/Circular)'),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopolyScenario(), 'agents': [(FixedPriceLEAgent, [])]},
		'the agent and marketplace must be of the same economy type (Linear/Circular)'),
	({'marketplace': linear_market.ClassicScenario(), 'agents': [(FixedPriceCEAgent, [])]},
		'the agent and marketplace must be of the same economy type (Linear/Circular)'),
	({'subfolder_name': 1}, 'subfolder_name must be of type str'),
	({'marketplace': linear_market.ClassicScenario(),
		'agents': [(QLearningCEAgent, ['ClassicScenario_QLearningLEAgent.dat'])]},
		'the agent and marketplace must be of the same economy type (Linear/Circular)'),
	({'marketplace': linear_market.MultiCompetitorScenario(),
		'agents': [(QLearningLEAgent,
		['CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'])]},
		'the modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyRebuyPriceMonopolyScenario(),
		'agents': [(actorcritic_agent.ContinuosActorCriticAgentFixedOneStd,
		['actor_parametersCircularEconomyRebuyPriceMonopolyScenario_ContinuosActorCriticAgentEstimatingStd.dat'])]},
		'the modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyRebuyPriceOneCompetitor(),
		'agents': [(actorcritic_agent.DiscreteACACircularEconomyRebuy,
		['actor_parametersCircularEconomyRebuyPriceOneCompetitor_ContinuosActorCriticAgentFixedOneStd.dat'])]},
		'the modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyMonopolyScenario(),
		'agents': [(actorcritic_agent.DiscreteACACircularEconomy,
		['actor_parametersCircularEconomyRebuyPriceOneCompetitor_DiscreteACACircularEconomyRebuy.dat'])]},
		'the modelfile is not compatible with the agent you tried to instantiate'),
	({'marketplace': circular_market.CircularEconomyMonopolyScenario(),
		'agents': [(QLearningCERebuyAgent,
		['CircularEconomyRebuyPriceOneCompetitor_QLearningCERebuyAgent.dat'])]},
		'the modelfile is not compatible with the agent you tried to instantiate')
]


@pytest.mark.parametrize('parameters, expected_message', incorrect_setup_monitoring_testcases)
def test_incorrect_setup_monitoring(parameters, expected_message):
	dict = {
		'enable_live_draw': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None,
		'subfolder_name': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	with pytest.raises(Exception) as assertion_message:
		monitor.configurator.setup_monitoring(
			enable_live_draw=dict['enable_live_draw'],
			episodes=dict['episodes'],
			plot_interval=dict['plot_interval'],
			marketplace=dict['marketplace'],
			agents=dict['agents'],
			subfolder_name=dict['subfolder_name']
		)
	assert expected_message in str(assertion_message.value)


incorrect_setup_monitoring_type_errors_testcases = [
	{'agents': [(linear_market.ClassicScenario(), [])]},
	{'agents': [(RuleBasedCEAgent(), [])]}
]


@pytest.mark.parametrize('parameters', incorrect_setup_monitoring_type_errors_testcases)
def test_incorrect_setup_monitoring_type_errors(parameters):
	dict = {
		'enable_live_draw': None,
		'episodes': None,
		'plot_interval': None,
		'marketplace': None,
		'agents': None,
		'subfolder_name': None
	}
	# replace the given parameters
	for key, val in parameters.items():
		dict[key] = val

	with pytest.raises(TypeError):
		monitor.configurator.setup_monitoring(
			enable_live_draw=dict['enable_live_draw'],
			episodes=dict['episodes'],
			plot_interval=dict['plot_interval'],
			marketplace=dict['marketplace'],
			agents=dict['agents'],
			subfolder_name=dict['subfolder_name']
		)


print_configuration_testcases = [
	([(RuleBasedCEAgent, [])]),
	([(RuleBasedCEAgent, []), (FixedPriceCEAgent, [])])
]


@pytest.mark.parametrize('agents', print_configuration_testcases)
def test_print_configuration(agents):
	monitor.configurator.setup_monitoring(agents=agents)

	monitor.configurator.print_configuration()


@pytest.mark.parametrize('agents', print_configuration_testcases)
def test_print_configuration_ratio(agents):
	monitor.configurator.setup_monitoring(episodes=51, plot_interval=1, agents=agents)

	with patch('recommerce.monitoring.agent_monitoring.am_configuration.input', create=True) as mocked_input:
		mocked_input.side_effect = ['n']
		monitor.configurator.print_configuration()
