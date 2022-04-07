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


# These testcases do not cover everything, nor should they, there are simply too many combinations
validate_config_valid_combined_not_final_testcases = [
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
	assert not validate_status
	assert 'If your config contains one of "environment" or "hyperparameter" it must also contain the other' == validate_data
##########
# End of tests with already combined configs (== hyperparameter and/or environment key on the top-level)
##########
