import os
from unittest.mock import patch

# from django.contrib.sessions.middleware import SessionMiddleware
# from django.test.client import RequestFactory
from django.test import TestCase

from ..handle_files import check_agents, download_file, handle_uploaded_file, parse_dict_to_database
from ..models.config import *


class MockedResponse():
	def __init__(self, header_content_disposition: str, file_for_content: str) -> None:
		self.headers = {'content-disposition': header_content_disposition}

		with open(file_for_content, 'rb') as file:
			self.content = file.read()


class MockedUploadedFile():
	def __init__(self, _name: str, _content: str) -> None:
		self.name = _name
		self.content = _content

	def chunks(self):
		return [self.content]


class FileHandling(TestCase):
	def test_right_zip_file_is_provided_for_download(self):
		path_to_tar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'example_tar_archive.tar')
		mocked_response = MockedResponse('filename=archive_results_Mar14_07-32-14.tar', path_to_tar)
		with patch('alpha_business_app.handle_files._add_files_to_zip'):
			response_file = download_file(mocked_response, True)

		assert 200 == response_file.status_code
		assert 'attachment; filename=archive_results_Mar14_07-32-14.zip' == response_file.headers['content-disposition']
		assert 'application/zip' == response_file.headers['content-type']

	def test_right_tar_file_is_provided_for_download(self):
		path_to_tar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'example_tar_archive.tar')
		mocked_response = MockedResponse('filename=archive_results_Mar14_07-32-14.tar', path_to_tar)
		with patch('alpha_business_app.handle_files._add_files_to_tar'):
			response_file = download_file(mocked_response, False)

		assert 200 == response_file.status_code
		assert 'attachment; filename=archive_results_Mar14_07-32-14.tar' == response_file.headers['content-disposition']
		assert 'application/tar' == response_file.headers['content-type']

	def test_uploaded_file_is_not_json(self):
		test_uploaded_file = MockedUploadedFile('test_file.jpg', b'this is a jpg')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('request', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'You can only upload files in JSON format.'} == actual_arguments[2]

	def test_uploaded_file_invalid_json(self):
		test_uploaded_file = MockedUploadedFile('test_file.json', b'{ "rl": "1234"')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('request', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'Your JSON is not valid'} == actual_arguments[2]

	def test_uploaded_file_with_unknown_key(self):
		test_uploaded_file = MockedUploadedFile('test_file.json', b'{ "test": "1234" }')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('request', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'The key test is unknown'} == actual_arguments[2]

	def test_objects_from_parse_dict(self):
		test_dict = {'rl': {'batch_size': 32}, 'sim_market': {'episode_size': 50}}
		resulting_config = parse_dict_to_database('hyperparameter', test_dict)

		assert resulting_config.sim_market is not None
		assert resulting_config.rl is not None

		# test all sim_market values
		sim_market_field_names = get_config_field_names(SimMarketConfig)
		for name in sim_market_field_names:
			if name != 'episode_size':
				assert getattr(resulting_config.sim_market, name) is None
			else:
				assert 50 == getattr(resulting_config.sim_market, name)

		# test all rl values
		rl_field_names = get_config_field_names(RlConfig)
		for name in rl_field_names:
			if name != 'batch_size':
				assert getattr(resulting_config.rl, name) is None
			else:
				assert 32 == getattr(resulting_config.rl, name)

	def test_parsing_with_only_hyperparameter(self):
		# get a test config to be parsed
		path_to_test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
		with open(os.path.join(path_to_test_data, 'test_hyperparameter_config.json'), 'r') as file:
			content = file.read()
		# mock uploaded file with test config
		test_uploaded_file = MockedUploadedFile('config.json', content.encode())
		# test method
		with patch('alpha_business_app.handle_files.redirect') as redirect_mock:
			handle_uploaded_file('this is not important', test_uploaded_file)
			redirect_mock.assert_called_once()
		# assert the datastructure, that should be present afterwards
		final_config: Config = Config.objects.all().first()
		assert Config == type(final_config)
		assert final_config.environment is None
		assert final_config.hyperparameter is not None

		hyperparameter_rl_config: RlConfig = final_config.hyperparameter.rl
		hyperparameter_sim_market_config: SimMarketConfig = final_config.hyperparameter.sim_market

		assert hyperparameter_rl_config is not None
		assert final_config.hyperparameter.sim_market is not None

		assert 0.99 == hyperparameter_rl_config.gamma
		assert 32 == hyperparameter_rl_config.batch_size
		assert 100000 == hyperparameter_rl_config.replay_size
		assert 1e-6 == hyperparameter_rl_config.learning_rate
		assert 1000 == hyperparameter_rl_config.sync_target_frames
		assert 10000 == hyperparameter_rl_config.replay_start_size
		assert 75000 == hyperparameter_rl_config.epsilon_decay_last_frame
		assert 1.0 == hyperparameter_rl_config.epsilon_start
		assert 0.1 == hyperparameter_rl_config.epsilon_final

		assert 100 == hyperparameter_sim_market_config.max_storage
		assert 50 == hyperparameter_sim_market_config.episode_size
		assert 10 == hyperparameter_sim_market_config.max_price
		assert 50 == hyperparameter_sim_market_config.max_quality
		assert 20 == hyperparameter_sim_market_config.number_of_customers
		assert 3 == hyperparameter_sim_market_config.production_price
		assert 0.1 == hyperparameter_sim_market_config.storage_cost_per_product

	def test_parsing_with_only_environment(self):
		# get a test config to be parsed
		path_to_test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
		with open(os.path.join(path_to_test_data, 'test_environment_config.json'), 'r') as file:
			content = file.read()
		# mock uploaded file with test config
		test_uploaded_file = MockedUploadedFile('config.json', content.encode())
		# test method
		with patch('alpha_business_app.handle_files.redirect') as redirect_mock:
			handle_uploaded_file('this is not important', test_uploaded_file)
			redirect_mock.assert_called_once()
		# assert the datastructure, that should be present afterwards
		final_config: Config = Config.objects.all().first()
		assert Config == type(final_config)
		assert final_config.environment is not None
		assert final_config.hyperparameter is None

		environment_config: EnvironmentConfig = final_config.environment

		assert 'agent_monitoring' == environment_config.task
		assert environment_config.enable_live_draw is False
		assert 50 == environment_config.episodes
		assert 25 == environment_config.plot_interval
		assert 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario' == environment_config.marketplace
		assert environment_config.agents is not None

		environment_agents: AgentsConfig = environment_config.agents

		all_rule_based_agents = list(environment_agents.rulebasedagentconfig_set.all())
		assert 1 == len(all_rule_based_agents)
		assert 'agents.vendors.RuleBasedCERebuyAgent' == all_rule_based_agents[0].agent_class
		assert all_rule_based_agents[0].argument is None

		all_qlearning_agents = list(environment_agents.cerebuyagentqlearningconfig_set.all())
		assert 1 == len(all_qlearning_agents)
		assert 'agents.vendors.QLearningCERebuyAgent' == all_qlearning_agents[0].agent_class
		assert 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat' == all_qlearning_agents[0].argument

	def test_parsing_mixed_config(self):
		# get a test config to be parsed
		path_to_test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
		with open(os.path.join(path_to_test_data, 'test_mixed_config.json'), 'r') as file:
			content = file.read()
		# mock uploaded file with test config
		test_uploaded_file = MockedUploadedFile('config.json', content.encode())
		# test method
		with patch('alpha_business_app.handle_files.redirect') as redirect_mock:
			handle_uploaded_file('this is not important', test_uploaded_file)
			redirect_mock.assert_called_once()
		# assert the datastructure, that should be present afterwards
		final_config: Config = Config.objects.all().first()
		assert Config == type(final_config)
		assert final_config.environment is not None
		assert final_config.hyperparameter is not None

		environment_config: EnvironmentConfig = final_config.environment
		hyperparameter_config: HyperparameterConfig = final_config.hyperparameter

		assert 'training' == environment_config.task
		assert environment_config.enable_live_draw is False
		assert 50 == environment_config.episodes

		assert hyperparameter_config.sim_market is not None
		assert hyperparameter_config.rl is not None

		assert 100 == hyperparameter_config.sim_market.max_storage
		assert 50 == hyperparameter_config.sim_market.episode_size
		assert 0.99 == hyperparameter_config.rl.gamma
		assert 32 == hyperparameter_config.rl.batch_size

	def test_parsing_invalid_rl_parameters(self):
		test_uploaded_file = MockedUploadedFile('config.json', b'{"rl": {"test":"bla"}}')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('this is not important', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'The keyword(s) [\'test\'] are not allowed in rl'} == actual_arguments[2]

	def test_parsing_duplicate_keys(self):
		test_uploaded_file = MockedUploadedFile('config.json', b'{"rl": {"test":"bla"}, "rl": {"test":"bla"}}')
		with patch('alpha_business_app.handle_files.render') as render_mock:
			handle_uploaded_file('this is not important', test_uploaded_file)

			actual_arguments = render_mock.call_args.args

			render_mock.assert_called_once()
			assert 'upload.html' == actual_arguments[1]
			assert {'error': 'Your config contains duplicate keys: \'rl\''} == actual_arguments[2]

	def test_valid_agents(self):
		status, error_msg = check_agents({'Rule_Based Agent': {'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'},
			'CE Rebuy Agent (QLearning)': {'agent_class': 'agents.vendors.QLearningCERebuyAgent',
				'argument': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'}})

		assert status is True
		assert '' == error_msg

	def test_invalid_agents(self):
		status, error_msg = check_agents({'Rule_Based Agent': {'test': 'agents.vendors.RuleBasedCERebuyAgent'}})

		assert status is False
		assert 'The keyword(s) [\'test\'] are not allowed in Rule_Based Agent' == error_msg

	# def test_invalid_values(self):
	# 	test_uploaded_file = MockedUploadedFile('config.json', b'{"rl": {"gamma": "bla"}')
	# 	with patch('alpha_business_app.handle_files.render') as render_mock:
	# 		handle_uploaded_file('this is not important', test_uploaded_file)

	# 		actual_arguments = render_mock.call_args.args

	# 		render_mock.assert_called_once()
	# 		assert 'upload.html' == actual_arguments[1]
	# 		assert {'error': ''} == actual_arguments[2]
