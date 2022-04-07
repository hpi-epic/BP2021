import pytest
import utils_tests as ut_t

import recommerce.configuration.config_validation as config_validation

validate_config_valid_final_testcases = [
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


@pytest.mark.parametrize('config', validate_config_valid_final_testcases)
def test_validate_config_valid_final(config):
	# If the config is valid, the first member of the tuple returned will be True
	validate_status, validate_data = config_validation.validate_config(config, True)
	assert validate_status, validate_data
