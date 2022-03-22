import json
from unittest.mock import mock_open, patch

import pytest
import utils_tests as ut_t

import recommerce.configuration.environment_config as env_config
from recommerce.market.circular.circular_sim_market import CircularEconomyRebuyPriceMonopolyScenario
from recommerce.market.circular.circular_vendors import RuleBasedCERebuyAgent
from recommerce.rl.q_learning.q_learning_agent import QLearningCERebuyAgent

valid_training_dict = {
	'task': 'training',
	'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'CE Rebuy Agent (QLearning)': {
			'class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent'
		}
	}
}

valid_agent_monitoring_dict = {
	'task': 'agent_monitoring',
	'enable_live_draw': False,
	'episodes': 10,
	'plot_interval': 5,
	'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'Rule_Based Agent': {
			'class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent'
		},
		'CE Rebuy Agent (QLearning)': {
			'class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
			'modelfile': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'
		}
	}
}

valid_exampleprinter_dict = {
	'task': 'exampleprinter',
	'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'CE Rebuy Agent (QLearning)': {
			'class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
			'modelfile': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'
		}
	}
}

invalid_agent_dict = {
	'task': 'exampleprinter',
	'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'Agent_name': {
			'class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
		}
	}
}

invalid_task_dict = {
	'task': 'not_existing_test_task',
	'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'Agent_name': {
			'class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
			'modelfile': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'
		}
	}
}


def test_abstract_parent_class():
	with pytest.raises(TypeError) as error_message:
		env_config.EnvironmentConfig()
	assert "Can't instantiate abstract class EnvironmentConfig" in str(error_message.value)


def test_str_representation():
	config = env_config.TrainingEnvironmentConfig(valid_training_dict)
	assert str(config) == "TrainingEnvironmentConfig: {'task': 'training', \
'marketplace': <class 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'>, \
'agent': <class 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent'>}"


get_class_testcases = [
	(CircularEconomyRebuyPriceMonopolyScenario,
		'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'),
	(QLearningCERebuyAgent, 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent'),
	(RuleBasedCERebuyAgent, 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent')
]


@pytest.mark.parametrize('expected_class, class_string', get_class_testcases)
def test_get_class(expected_class, class_string):
	assert expected_class == env_config.EnvironmentConfig._get_class(env_config.EnvironmentConfig, class_string)


def test_get_class_invalid_class():
	with pytest.raises(AttributeError) as error_message:
		env_config.EnvironmentConfig._get_class(env_config.EnvironmentConfig, 'recommerce.market.circular.circular_vendors.NotAValidClass')
	assert 'The string you passed could not be resolved to a class' in str(error_message.value)


def test_get_class_invalid_module():
	with pytest.raises(ModuleNotFoundError) as error_message:
		env_config.EnvironmentConfig._get_class(env_config.EnvironmentConfig, 'notAModule.ValidClass')
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
	valid_training_dict,
	valid_agent_monitoring_dict,
	valid_exampleprinter_dict
]


@pytest.mark.parametrize('config', valid_ConfigLoader_validate_testcases)
def test_valid_ConfigLoader_validate(config):
	env_config.EnvironmentConfigLoader.validate(config)


valid_ConfigLoader_load_training_testcases = [
	# TODO: Currently no testcases for ActorCriticAgents
	('training', 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
		{'CE Rebuy Agent (QLearning)': {'class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent'}}),
	('training', 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceOneCompetitor',
		{'CE Rebuy Agent (QLearning)': {'class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCEAgent'}})
]


@pytest.mark.parametrize('task, marketplace, agents', valid_ConfigLoader_load_training_testcases)
def test_valid_ConfigLoader_load_training(task, marketplace, agents):
	mock_json = json.dumps(ut_t.create_environment_mock_dict(task=task, marketplace=marketplace, agents=agents))
	with patch('builtins.open', mock_open(read_data=mock_json)) as mock_file:
		ut_t.check_mock_file(mock_file, mock_json)
		env_config.EnvironmentConfigLoader.load('environment_config_training.json')


configLoader_is_valid_testcases = [
	(invalid_agent_dict, False, 'This agent must have a "modelfile" field'),
	(invalid_task_dict, False, 'The specified task is unknown: not_existing_test_task'),
	(valid_exampleprinter_dict, True, 'Your config is valid.')
]


@pytest.mark.parametrize('config, expected_status, expected_error', configLoader_is_valid_testcases)
def test_is_valid(config, expected_status, expected_error):
	status, error = env_config.EnvironmentConfigLoader.is_valid(config)
	assert expected_status == status
	assert error.startswith(expected_error)