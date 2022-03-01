import pytest

import configuration.environment_config as env_config
from agents.vendors import QLearningCERebuyAgent, RuleBasedCERebuyAgent
from market.circular.circular_sim_market import CircularEconomyRebuyPriceMonopolyScenario

valid_training_dict = {
	'task': 'training',
	'marketplace': 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'Agent_name': {
			'class': 'agents.vendors.QLearningCERebuyAgent'
		}
	}
}

valid_agent_monitoring_dict = {
	'task': 'agent_monitoring',
	'enable_live_draw': False,
	'episodes': 10,
	'plot_interval': 5,
	'marketplace': 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'Rule_Based Agent': {
			'class': 'agents.vendors.RuleBasedCERebuyAgent'
		},
		'CE Rebuy Agent (QLearning)': {
			'class': 'agents.vendors.QLearningCERebuyAgent',
			'modelfile': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'
		}
	}
}

valid_exampleprinter_dict = {
	'task': 'exampleprinter',
	'marketplace': 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
	'agents': {
		'Agent_name': {
			'class': 'agents.vendors.QLearningCERebuyAgent',
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
'marketplace': <class 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'>, \
'agent': <class 'agents.vendors.QLearningCERebuyAgent'>}"


get_class_testcases = [
	(CircularEconomyRebuyPriceMonopolyScenario, 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'),
	(QLearningCERebuyAgent, 'agents.vendors.QLearningCERebuyAgent'),
	(RuleBasedCERebuyAgent, 'agents.vendors.RuleBasedCERebuyAgent')
]


@pytest.mark.parametrize('expected_class, class_string', get_class_testcases)
def test_get_class(expected_class, class_string):
	assert expected_class == env_config.EnvironmentConfig._get_class(env_config.EnvironmentConfig, class_string)


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