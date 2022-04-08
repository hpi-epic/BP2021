import pytest
import utils_tests as ut_t

import recommerce.configuration.config_validation as config_validation

##########
# Tests with already combined configs (== hyperparameter and/or environment key on the top-level)
##########
validate_config_valid_combined_final_testcases = [
	ut_t.create_combined_mock_dict(),
	ut_t.create_combined_mock_dict(hyperparameter=ut_t.create_hyperparameter_mock_dict(rl=ut_t.create_hyperparameter_mock_dict_rl(gamma=0.5))),
	ut_t.create_combined_mock_dict(hyperparameter=ut_t.create_hyperparameter_mock_dict(
		sim_market=ut_t.create_hyperparameter_mock_dict_sim_market(max_price=25))),
	ut_t.create_combined_mock_dict(environment=ut_t.create_environment_mock_dict(task='exampleprinter')),
	ut_t.create_combined_mock_dict(environment=ut_t.create_environment_mock_dict(agents=[
		{
			'name': 'Test_agent',
			'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
			'argument': ''
		},
		{
			'name': 'Test_agent2',
			'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
			'argument': ''
		}
	])),
]


@pytest.mark.parametrize('config', validate_config_valid_combined_final_testcases)
def test_validate_config_valid_combined_final(config):
	# If the config is valid, the first member of the tuple returned will be True
	validate_status, validate_data = config_validation.validate_config(config, True)
	assert validate_status, validate_data
	assert isinstance(validate_data, tuple)
	assert 'rl' in validate_data[0]
	assert 'sim_market' in validate_data[0]
	assert 'gamma' in validate_data[0]['rl']
	assert 'max_price' in validate_data[0]['sim_market']
	assert 'task' in validate_data[1]
	assert 'agents' in validate_data[1]


# These testcases do not cover everything, nor should they, there are simply too many combinations
validate_config_valid_combined_not_final_testcases = [
	ut_t.create_combined_mock_dict(
		hyperparameter=ut_t.remove_key('rl', ut_t.create_hyperparameter_mock_dict())),
	ut_t.create_combined_mock_dict(
		hyperparameter=ut_t.create_hyperparameter_mock_dict(
			rl=ut_t.remove_key('learning_rate', ut_t.create_hyperparameter_mock_dict_rl(gamma=0.5)))),
	ut_t.create_combined_mock_dict(
		hyperparameter=ut_t.create_hyperparameter_mock_dict(
			rl=ut_t.remove_key('epsilon_start', ut_t.remove_key('learning_rate', ut_t.create_hyperparameter_mock_dict_rl())))),
	ut_t.create_combined_mock_dict(environment=ut_t.remove_key('task', ut_t.create_environment_mock_dict())),
	ut_t.create_combined_mock_dict(environment=ut_t.remove_key('agents', ut_t.remove_key('task', ut_t.create_environment_mock_dict()))),
] + validate_config_valid_combined_final_testcases


@pytest.mark.parametrize('config', validate_config_valid_combined_not_final_testcases)
def test_validate_config_valid_combined_not_final(config):
	# If the config is valid, the first member of the returned tuple will be True
	validate_status, validate_data = config_validation.validate_config(config, False)
	assert validate_status, validate_data


test_validate_config_one_top_key_missing_testcases = [
	(ut_t.create_combined_mock_dict(hyperparameter=None), True),
	(ut_t.create_combined_mock_dict(environment=None), True),
	(ut_t.create_combined_mock_dict(hyperparameter=None), False),
	(ut_t.create_combined_mock_dict(environment=None), False)
]


@pytest.mark.parametrize('config, is_final', test_validate_config_one_top_key_missing_testcases)
def test_validate_config_one_top_key_missing(config, is_final):
	validate_status, validate_data = config_validation.validate_config(config, is_final)
	assert not validate_status, validate_data
	assert 'If your config contains one of "environment" or "hyperparameter" it must also contain the other' == validate_data


test_validate_config_too_many_keys_testcases = [
	True,
	False
]


@pytest.mark.parametrize('is_final', test_validate_config_too_many_keys_testcases)
def test_validate_config_too_many_keys(is_final):
	test_config = ut_t.create_combined_mock_dict()
	test_config['additional_key'] = "this should'nt be allowed"
	validate_status, validate_data = config_validation.validate_config(test_config, is_final)
	assert not validate_status, validate_data
	assert 'Your config should not contain keys other than "environment" and "hyperparameter"' == validate_data
##########
# End of tests with already combined configs (== hyperparameter and/or environment key on the top-level)
##########


##########
# Tests without the already split top-level (config keys are mixed and need to be matched)
##########
# These are singular dicts that will get combined for the actual testcases
test_validate_config_valid_not_final_dicts = [
	{
		'rl': {
			'gamma': 0.5,
			'epsilon_start': 0.9
		}
	},
	{
		'sim_market': {
			'max_price': 40
		}
	},
	{
		'task': 'training'
	},
	{
		'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'
	},
	{
		'agents': [
			{
				'name': 'Rule_Based Agent',
				'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
				'argument': ''
			},
			{
				'name': 'CE Rebuy Agent (QLearning)',
				'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
				'argument': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'
			}
		]
	},
	{
		'agents': [
			{
				'name': 'Rule_Based Agent',
				'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
				'argument': ''
			}
		]
	}
]


# get all combinations of the dicts defined above to mix and match as much as possible
test_validate_config_valid_not_final_testcases = [
	{**dict1, **dict2} for dict1 in test_validate_config_valid_not_final_dicts for dict2 in test_validate_config_valid_not_final_dicts
]


@pytest.mark.parametrize('config', test_validate_config_valid_not_final_testcases)
def test_validate_config_valid_not_final(config):
	validate_status, validate_data = config_validation.validate_config(config, False)
	assert validate_status, f'Test failed with error: {validate_data} on config: {config}'


test_validate_config_valid_final_testcases = [
	{**ut_t.create_hyperparameter_mock_dict(), **ut_t.create_environment_mock_dict()},
	{**ut_t.create_hyperparameter_mock_dict(rl=ut_t.create_hyperparameter_mock_dict_rl(gamma=0.2)), **ut_t.create_environment_mock_dict()},
	{**ut_t.create_hyperparameter_mock_dict(), **ut_t.create_environment_mock_dict(episodes=20)}
]


@pytest.mark.parametrize('config', test_validate_config_valid_final_testcases)
def test_validate_config_valid_final(config):
	validate_status, validate_data = config_validation.validate_config(config, True)
	assert validate_status, f'Test failed with error: {validate_data} on config: {config}'
	assert 'rl' in validate_data[0]
	assert 'sim_market' in validate_data[0]
	assert 'agents' in validate_data[1]
