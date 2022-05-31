from django.test import TestCase

from ..models.agent_config import AgentConfig
from ..models.agents_config import AgentsConfig
from ..models.config import Config
from ..models.container import Container
from ..models.environment_config import EnvironmentConfig
from ..models.rl_config import RlConfig
from ..utils import capitalize, remove_none_values_from_dict, to_config_class_name

# from .constant_tests import EMPTY_STRUCTURE_CONFIG


class ConfigTest(TestCase):
	def test_class_name_config(self):
		assert 'Config' == to_config_class_name('')

	def test_class_name_environment_config(self):
		assert 'EnvironmentConfig' == to_config_class_name('environment')

	def test_class_name_agents_config(self):
		assert 'AgentsConfig' == to_config_class_name('agents')

	def test_class_name_hyperparameter_config(self):
		assert 'HyperparameterConfig' == to_config_class_name('hyperparameter')

	def test_class_name_rl_config(self):
		assert 'RlConfig' == to_config_class_name('rl')

	def test_class_name_sim_market_config(self):
		assert 'SimMarketConfig' == to_config_class_name('sim_market')

	def test_capitalize(self):
		assert 'TestTesTTest' == capitalize('testTesTTest')

	def test_capitalize_empty_strings(self):
		assert '' == capitalize('')

	def test_capitalize_one_letter_strings(self):
		assert 'A' == capitalize('a')

	def test_is_referenced(self):
		test_config_not_referenced = Config.objects.create()
		test_config_referenced = Config.objects.create()

		from django.contrib.auth.models import User
		user = User.objects.create(username='test_user', password='top_secret')
		Container.objects.create(config=test_config_referenced, user=user)

		assert test_config_not_referenced.is_referenced() is False
		assert test_config_referenced.is_referenced() is True

	# def test_config_to_dict(self):
	# 	# create a small valid config for this test
	# 	agents_config = AgentsConfig.objects.create()

	# 	AgentConfig.objects.create(agent_class='recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
	# 		argument='', agents_config=agents_config, name='Rule_Based Agent')

	# 	env_config = EnvironmentConfig.objects.create(agents=agents_config,
	# 		marketplace='recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopoly',
	# 		task='training')

	# 	rl_config = RlConfig.objects.create(gamma=0.99,
	# 		batch_size=32,
	# 		replay_size=100000,
	# 		learning_rate=1e-6,
	# 		sync_target_frames=1000,
	# 		replay_start_size=10000,
	# 		epsilon_decay_last_frame=75000,
	# 		epsilon_start=1.0,
	# 		epsilon_final=0.1)

	# 	sim_market_config = SimMarketConfig.objects.create(max_storage=100,
	# 		episode_length=50,
	# 		max_price=10,
	# 		max_quality=50,
	# 		number_of_customers=20,
	# 		production_price=3,
	# 		storage_cost_per_product=0.1)

	# 	hyperparameter_config = HyperparameterConfig.objects.create(sim_market=sim_market_config,
	# 		rl=rl_config)

	# 	final_config = Config.objects.create(environment=env_config,
	# 		hyperparameter=hyperparameter_config)
	# 	expected_dict = {
	# 			'hyperparameter': {
	# 				'rl': {
	# 					'gamma': 0.99,
	# 					'batch_size': 32,
	# 					'replay_size': 100000,
	# 					'learning_rate': 1e-6,
	# 					'sync_target_frames': 1000,
	# 					'replay_start_size': 10000,
	# 					'epsilon_decay_last_frame': 75000,
	# 					'epsilon_start': 1.0,
	# 					'epsilon_final': 0.1
	# 				},
	# 				'sim_market': {
	# 					'max_storage': 100,
	# 					'episode_length': 50,
	# 					'max_price': 10,
	# 					'max_quality': 50,
	# 					'number_of_customers': 20,
	# 					'production_price': 3,
	# 					'storage_cost_per_product': 0.1
	# 				}
	# 			},
	# 			'environment': {
	# 				'task': 'training',
	# 				'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopoly',
	# 				'agents': [
	# 					{
	# 						'name': 'Rule_Based Agent',
	# 						'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
	# 						'argument': ''
	# 					}
	# 				]
	# 			}
	# 		}
	# 	assert expected_dict == final_config.as_dict()

	def test_dict_representation_of_agent(self):
		test_agent = AgentConfig.objects.create(name='test_agent', agent_class='test_class', argument='1234')
		expected_dict = {'name': 'test_agent', 'agent_class': 'test_class', 'argument': '1234'}
		assert expected_dict == test_agent.as_dict()

	def test_dict_representation_of_environment_config(self):
		test_environment = EnvironmentConfig.objects.create(enable_live_draw=True, episodes=50, plot_interval=12, marketplace='test')
		expected_dict = {'enable_live_draw': True, 'episodes': 50, 'plot_interval': 12, 'marketplace': 'test'}
		assert expected_dict == test_environment.as_dict()

	def test_dict_representation_of_empty_config(self):
		test_config = Config.objects.create()
		assert {} == test_config.as_dict()

	def test_list_representation_of_agents(self):
		test_agents = AgentsConfig.objects.create()
		AgentConfig.objects.create(name='test_agent1', agent_class='test_class', agents_config=test_agents)
		AgentConfig.objects.create(name='test_agent2', agent_class='test_class', agents_config=test_agents)

		expected_list = [
				{
					'name': 'test_agent1',
					'agent_class': 'test_class',
					'argument': ''
				},
				{
					'name': 'test_agent2',
					'agent_class': 'test_class',
					'argument': ''
				}
			]
		assert expected_list == test_agents.as_list()

	# def test_get_empty_structure_dict(self):
	# 	actual_dict = Config.get_empty_structure_dict()
	# 	assert EMPTY_STRUCTURE_CONFIG == actual_dict

	def test_get_empty_structure_dict_for_rl(self):
		expected_dict = {}
		assert expected_dict == RlConfig.get_empty_structure_dict()

	def test_remove_none_values_from_dict(self):
		test_dict = {'test': 'test', 'test2': None}
		assert {'test': 'test'} == remove_none_values_from_dict(test_dict)
