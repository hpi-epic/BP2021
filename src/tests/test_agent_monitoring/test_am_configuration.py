import os
import re
import shutil
from unittest.mock import patch

import pytest

import agents.vendors as vendors
# import market.sim_market as sim_market
import market.linear_market.linear_sim_market as linear_sim_market
import market.circular_market.circular_sim_market as circular_sim_market
import monitoring.agent_monitoring.am_configuration as am_configuration
import monitoring.agent_monitoring.am_monitoring as monitoring

monitor = monitoring.Monitor()


# setup before each test
def setup_function(function):
	print('***SETUP***')
	global monitor
	monitor = monitoring.Monitor()
	monitor.configurator.setup_monitoring(enable_live_draw=False, subfolder_name=f'test_plots_{function.__name__}')


def teardown_module(module):
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'results', 'monitoring')):
		if re.match('test_*', file_name):
			assert False, 'file writing was not mocked or a created file was not removed after the test!'


def test_init_default_values():
	test_configurator = am_configuration.Configurator()
	assert test_configurator.enable_live_draw is True
	assert 500 == test_configurator.episodes
	assert 50 == test_configurator.plot_interval
	assert isinstance(test_configurator.marketplace, circular_sim_market.CircularEconomyMonopolyScenario)
	assert isinstance(test_configurator.agents[0], vendors.QLearningCEAgent)
	assert 1 == len(test_configurator.agents)
	assert [(0.0, 0.0, 1.0, 1.0)] == test_configurator.agent_colors
	# folder_path can hardly be tested due to the default involving the current DateTime


def test_get_folder():
	# if you change the name of this function, change it here as well!
	foldername = 'test_plots_test_get_folder'
	monitor.configurator.get_folder()
	assert os.path.exists(
		os.path.abspath(os.path.join(os.path.dirname(__file__),
		os.pardir, os.pardir, os.pardir,
		'results', 'monitoring', foldername)))
	shutil.rmtree(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'results', 'monitoring', foldername))


def test_get_modelfile_path():
	with patch('monitoring.agent_monitoring.am_configuration.os.path.exists') as mock_exists:
		mock_exists.return_value = False
		with pytest.raises(AssertionError) as assertion_message:
			monitor.configurator._get_modelfile_path('non_existing_modelfile')
		assert 'the specified modelfile does not exist' in str(assertion_message.value)


incorrect_update_agents_testcases = [
	([(vendors.QLearningCEAgent, ['modelfile.dat', 'arg', 'too_much'])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(vendors.QLearningCEAgent, [1, 2, 3, 4])], 'the argument list for a RL-agent must have length between 0 and 2'),
	([(vendors.QLearningCEAgent, ['modelfile.dat', 35])], 'the arguments for a RL-agent must be of type str'),
	([(vendors.QLearningCEAgent, [25])], 'the arguments for a RL-agent must be of type str'),
	([(vendors.QLearningCEAgent, ['agent_name', 'modelfile.dat'])], 'if two arguments are provided, the first one must be the modelfile.'),
	([(vendors.QLearningCEAgent, ['mymodel.dat'])], 'the specified modelfile does not exist')
]


@pytest.mark.parametrize('agents, expected_message', incorrect_update_agents_testcases)
def test_incorrect_update_agents(agents, expected_message):
	with pytest.raises(AssertionError) as assertion_message:
		monitor.configurator.setup_monitoring(agents=agents)
	assert expected_message in str(assertion_message.value)


correct_update_agents_testcases = [
	[(vendors.QLearningCEAgent, [])],
	[(vendors.QLearningCEAgent, ['new_name'])],
	[(vendors.QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat'])],
	[(vendors.QLearningCEAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'new_name'])],
	[(vendors.QLearningCEAgent, [f'{type(monitor.configurator.marketplace).__name__}_{vendors.QLearningCEAgent.__name__}.dat'])]
]


@pytest.mark.parametrize('agents', correct_update_agents_testcases)
def test_correct_update_agents(agents):
	monitor.configurator.setup_monitoring(agents=agents)


def test_correct_setup_monitoring():
	monitor.configurator.setup_monitoring(
		enable_live_draw=False,
		episodes=10,
		plot_interval=2,
		marketplace=circular_sim_market.CircularEconomyMonopolyScenario,
		agents=[(vendors.HumanPlayerCERebuy, ['reptiloid']),
			(vendors.QLearningCERebuyAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'q_learner'])],
		subfolder_name='subfoldername')
	assert monitor.configurator.enable_live_draw is False
	assert 10 == monitor.configurator.episodes
	assert 2 == monitor.configurator.plot_interval
	assert isinstance(monitor.configurator.marketplace, circular_sim_market.CircularEconomyMonopolyScenario)
	assert 2 == len(monitor.configurator.agents)
	assert isinstance(monitor.configurator.agents[0], vendors.HumanPlayerCERebuy)
	assert isinstance(monitor.configurator.agents[1], vendors.QLearningCERebuyAgent)
	assert 'reptiloid' == monitor.configurator.agents[0].name
	assert 'q_learner' == monitor.configurator.agents[1].name
	assert os.path.normcase(
		os.path.abspath(
			os.path.join(
				os.path.dirname(__file__),
				os.pardir, os.pardir, os.pardir,
				'results', 'monitoring', 'subfoldername'
			)
		)
	) == os.path.normcase(os.path.abspath(monitor.configurator.folder_path))
	assert 2 == len(monitor.configurator.agent_colors)


setting_multiple_agents_testcases = [
	[(vendors.FixedPriceCERebuyAgent, []), (vendors.FixedPriceCEAgent, [])],
	[(vendors.FixedPriceCERebuyAgent, []), (vendors.FixedPriceCEAgent, []), (vendors.FixedPriceCERebuyAgent, [])]
]


@pytest.mark.parametrize('agents', setting_multiple_agents_testcases)
def test_setting_multiple_agents(agents):
	monitor.configurator.setup_monitoring(agents=agents)


def test_setting_market_not_agents():
	monitor.configurator.setup_monitoring(marketplace=circular_sim_market.CircularEconomyMonopolyScenario)


incorrect_setup_monitoring_testcases = [
	({'enable_live_draw': 1}, 'enable_live_draw must be a Boolean'),
	({'episodes': 'Hello World'}, 'episodes must be of type int'),
	({'episodes': 0}, 'episodes must not be 0'),
	({'plot_interval': '1'}, 'plot_interval must be of type int'),
	({'plot_interval': 0}, 'plot_interval must not be 0'),
	({'episodes': 4, 'plot_interval': 5}, 'plot_interval must be <= episodes, or no plots can be generated.'),
	({'marketplace': vendors.RuleBasedCEAgent}, 'the marketplace must be a subclass of SimMarket'),
	({'marketplace': linear_sim_market.ClassicScenario, 'agents': [(vendors.RuleBasedCEAgent, [])]},
		'the agent and marketplace must be of the same economy type'),
	({'agents': [vendors.RuleBasedCEAgent]}, 'agents must be a list of tuples'),
	({'agents': [[vendors.RuleBasedCEAgent, 1, '2']]}, 'agents must be a list of tuples'),
	({'agents': [(vendors.RuleBasedCEAgent)]}, 'agents must be a list of tuples'),
	({'agents': [vendors.FixedPriceLEAgent, vendors.FixedPriceCERebuyAgent]}, 'agents must be a list of tuples'),
	({'agents': [(vendors.RuleBasedCEAgent, ['arg'], 'too_much')]},
		'the list entries in agents must have size 2 ([agent_class, arguments])'),
	({'agents': [(linear_sim_market.ClassicScenario, [])]}, 'the first entry in each agent-tuple must be an agent class in `vendors.py`'),
	({'agents': [(vendors.RuleBasedCEAgent, linear_sim_market.ClassicScenario)]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(vendors.RuleBasedCEAgent, 'new_name')]}, 'the second entry in each agent-tuple must be a list'),
	({'agents': [(vendors.RuleBasedCEAgent, []), (vendors.FixedPriceLEAgent, [])]},
		'the agents must all be of the same type (Linear/Circular)'),
	({'agents': [(vendors.RuleBasedCEAgent, []), (vendors.FixedPriceLEAgent, []), (vendors.FixedPriceCEAgent, [])]},
		'the agents must all be of the same type (Linear/Circular)'),
	({'marketplace': circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario, 'agents': [(vendors.FixedPriceLEAgent, [])]},
		'the agent and marketplace must be of the same economy type (Linear/Circular)'),
	({'marketplace': linear_sim_market.ClassicScenario, 'agents': [(vendors.FixedPriceCEAgent, [])]},
		'the agent and marketplace must be of the same economy type (Linear/Circular)'),
	({'subfolder_name': 1}, 'subfolder_name must be of type str')
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

	with pytest.raises(AssertionError) as assertion_message:
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
	{'marketplace': linear_sim_market.ClassicScenario()},
	{'agents': [(linear_sim_market.ClassicScenario(), [])]},
	{'agents': [(vendors.RuleBasedCEAgent(), [])]}
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


def test_get_configuration():
	monitor.configurator.setup_monitoring(
		enable_live_draw=False,
		episodes=10,
		plot_interval=2,
		marketplace=circular_sim_market.CircularEconomyMonopolyScenario,
		agents=[(vendors.HumanPlayerCERebuy, ['reptiloid']),
			(vendors.QLearningCERebuyAgent, ['CircularEconomyMonopolyScenario_QLearningCEAgent.dat', 'q_learner'])],
		subfolder_name='subfoldername')
	current_configuration = monitor.configurator.get_configuration()
	assert len(current_configuration) == 7, 'parameters were updated in agent_monitoring.py, but not updated in the tests!'
	assert 'enable_live_draw' in current_configuration
	assert 'episodes' in current_configuration
	assert 'plot_interval' in current_configuration
	assert 'marketplace' in current_configuration
	assert 'agents' in current_configuration
	assert 'agent_colors' in current_configuration
	assert 'folder_path' in current_configuration


print_configuration_testcases = [
	([(vendors.RuleBasedCEAgent, [])]),
	([(vendors.RuleBasedCEAgent, []), (vendors.FixedPriceCEAgent, [])])
]


@pytest.mark.parametrize('agents', print_configuration_testcases)
def test_print_configuration(agents):
	monitor.configurator.setup_monitoring(agents=agents)

	monitor.configurator.print_configuration()


@pytest.mark.parametrize('agents', print_configuration_testcases)
def test_print_configuration_ratio(agents):
	monitor.configurator.setup_monitoring(episodes=51, plot_interval=1, agents=agents)

	with patch('monitoring.agent_monitoring.am_configuration.input', create=True) as mocked_input:
		mocked_input.side_effect = ['n']
		monitor.configurator.print_configuration()
