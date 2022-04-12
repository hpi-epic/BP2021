import json
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import recommerce.configuration.environment_config as env_config
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceMonopolyScenario
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningAgent


def test_abstract_parent_class():
	with pytest.raises(TypeError) as error_message:
		env_config.EnvironmentConfig()
	assert "Can't instantiate abstract class EnvironmentConfig" in str(error_message.value)


def test_get_required_fields_valid():
	fields = env_config.EnvironmentConfig.get_required_fields('top-dict')
	assert fields == {
				'task': False,
				'enable_live_draw': False,
				'episodes': False,
				'plot_interval': False,
				'marketplace': False,
				'agents': False
			}


def test_get_required_fields_invalid():
	with pytest.raises(AssertionError) as error_message:
		env_config.EnvironmentConfig.get_required_fields('wrong_key')
	assert 'The given level does not exist in an environment-config: wrong_key' in str(error_message.value)


def test_str_representation():
	test_dict = {
		'task': 'training',
		'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
		'agents': [
			{
				'name': 'CE Rebuy Agent (QLearning)',
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent',
				'argument': ''
			}
		]
	}
	config = env_config.TrainingEnvironmentConfig(test_dict)
	assert str(config) == ("TrainingEnvironmentConfig: {'task': 'training', "
		"'marketplace': <class 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'>, "
		"'agent': {'name': 'CE Rebuy Agent (QLearning)', "
		"'agent_class': <class 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent'>, 'argument': ''}}")


get_class_testcases = [
	(CircularEconomyRebuyPriceMonopolyScenario,
		'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'),
	(QLearningAgent, 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent'),
	(RuleBasedCERebuyAgent, 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent')
]


@pytest.mark.parametrize('expected_class, class_string', get_class_testcases)
def test_get_class(expected_class, class_string):
	assert expected_class == env_config.get_class(class_string)


def test_get_class_invalid_class():
	with pytest.raises(AttributeError) as error_message:
		env_config.get_class('recommerce.market.circular.circular_vendors.NotAValidClass')
	assert 'The string you passed could not be resolved to a class' in str(error_message.value)


instantiate_invalid_get_class_return_testcases = [
	('recommerce.market.circular.circular_sim_market.CircularEconomy', 'CircularEconomy'),
	('recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPrice', 'CircularEconomyRebuyPrice'),
	('recommerce.market.sim_market.SimMarket', 'SimMarket'),
	('recommerce.market.linear.linear_sim_market.LinearEconomy', 'LinearEconomy')
]


@pytest.mark.parametrize('complete_class_string, splitted_class_string', instantiate_invalid_get_class_return_testcases)
def test_instantiate_invalid_get_class_return(complete_class_string, splitted_class_string):
	with pytest.raises(TypeError) as error_message:
		env_config.get_class(complete_class_string)()
	assert str(error_message.value).startswith(f'Can\'t instantiate abstract class {splitted_class_string} with abstract methods')


def test_get_class_invalid_module():
	with pytest.raises(ModuleNotFoundError) as error_message:
		env_config.get_class('notAModule.ValidClass')
	assert 'The string you passed could not be resolved to a module' in str(error_message.value)


get_task_testcases = [
	(env_config.TrainingEnvironmentConfig, 'training'),
	(env_config.AgentMonitoringEnvironmentConfig, 'agent_monitoring'),
	(env_config.ExampleprinterEnvironmentConfig, 'exampleprinter')
]


@pytest.mark.parametrize('tested_class, expected_task', get_task_testcases)
def test_get_task(tested_class: env_config.EnvironmentConfig, expected_task):
	assert expected_task == tested_class._get_task(tested_class)


valid_ConfigLoader_validate_testcases = [
	{
		'task': 'training',
		'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
		'agents': [
			{
				'name': 'CE Rebuy Agent (QLearning)',
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent',
				'argument': ''
			}
		]
	},
	{
		'task': 'agent_monitoring',
		'enable_live_draw': False,
		'episodes': 10,
		'plot_interval': 5,
		'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
		'agents': [
			{
				'name': 'Rule_Based Agent',
				'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
				'argument': ''
			},
			{
				'name': 'CE Rebuy Agent (QLearning)',
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent',
				'argument': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningAgent.dat'
			}
		]
	},
	{
		'task': 'exampleprinter',
		'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
		'agents': [
			{
				'name': 'CE Rebuy Agent (QLearning)',
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent',
				'argument': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningAgent.dat'
			}
		]
	}
]


@pytest.mark.parametrize('config', valid_ConfigLoader_validate_testcases)
def test_valid_ConfigLoader_validate(config):
	env_config.EnvironmentConfigLoader.validate(config)


valid_ConfigLoader_load_training_testcases = [
	# TODO: Currently no testcases for ActorCriticAgents
	('training', 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
		[{'name': 'CE Rebuy Agent (QLearning)', 'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent',
			'argument': ''}]),
	('training', 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceOneCompetitor',
		[{'name': 'CE Rebuy Agent (QLearning)', 'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent', 'argument': ''}])
]


@pytest.mark.parametrize('task, marketplace, agents', valid_ConfigLoader_load_training_testcases)
def test_valid_ConfigLoader_load_training(task, marketplace, agents):
	mock_json = json.dumps(ut_t.create_environment_mock_dict(task=task, marketplace=marketplace, agents=agents))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		env_config.EnvironmentConfigLoader.load('environment_config_training.json')
