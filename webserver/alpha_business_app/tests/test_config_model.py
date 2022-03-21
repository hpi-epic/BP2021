from django.test import TestCase

from ..models.config import *
from ..models.config import get_config_field_names, remove_none_values_from_dict


class ConfigTest(TestCase):
	def test_right_field_names_config(self):
		expected_field_names = ['environment', 'hyperparameter']
		actual_field_names = get_config_field_names(Config)
		assert expected_field_names == actual_field_names

	def test_right_field_names_environment_config(self):
		expected_field_names = ['agents', 'enable_live_draw', 'episodes', 'plot_interval', 'marketplace', 'task']
		actual_field_names = get_config_field_names(EnvironmentConfig)
		assert expected_field_names == actual_field_names

	def test_right_field_names_agents_config(self):
		expected_field_names = []
		actual_field_names = get_config_field_names(AgentsConfig)
		assert expected_field_names == actual_field_names

	def test_right_field_names_hyperparameter_config(self):
		expected_field_names = ['rl', 'sim_market']
		actual_field_names = get_config_field_names(HyperparameterConfig)
		assert expected_field_names == actual_field_names

	def test_right_field_names_rl_config(self):
		expected_field_names = ['gamma', 'batch_size', 'replay_size', 'learning_rate', 'sync_target_frames', 'replay_start_size',
			'epsilon_decay_last_frame', 'epsilon_start', 'epsilon_final']
		actual_field_names = get_config_field_names(RlConfig)
		assert expected_field_names == actual_field_names

	def test_right_field_names_sim_market_config(self):
		expected_field_names = ['max_storage', 'episode_size', 'max_price', 'max_quality', 'number_of_customers', 'production_price',
			'storage_cost_per_product']
		actual_field_names = get_config_field_names(SimMarketConfig)
		assert expected_field_names == actual_field_names

	def test_class_name_config(self):
		assert 'Config' == to_config_class_name('')

	def test_class_name_environment_config(self):
		assert 'EnvironmentConfig' == to_config_class_name('environment')

	def test_class_name_agents_config(self):
		assert 'AgentsConfig' == to_config_class_name('agents')

	def test_class_name_rule_based_config(self):
		assert 'RuleBasedAgentConfig' == to_config_class_name('Rule_Based Agent')

	def test_class_name_q_learing_config(self):
		assert 'CERebuyAgentQLearningConfig' == to_config_class_name('CE Rebuy Agent (QLearning)')

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

	def test_config_to_dict(self):
		# create a small valid config for this test
		agents_config = AgentsConfig.objects.create()

		RuleBasedAgentConfig.objects.create(agent_class='agents.vendors.RuleBasedCERebuyAgent',
			agents_config=agents_config)

		env_config = EnvironmentConfig.objects.create(agents=agents_config,
			marketplace='market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
			task='training')

		rl_config = RlConfig.objects.create(gamma=0.99,
			batch_size=32,
			replay_size=100000,
			learning_rate=1e-6,
			sync_target_frames=1000,
			replay_start_size=10000,
			epsilon_decay_last_frame=75000,
			epsilon_start=1.0,
			epsilon_final=0.1)

		sim_market_config = SimMarketConfig.objects.create(max_storage=100,
			episode_size=50,
			max_price=10,
			max_quality=50,
			number_of_customers=20,
			production_price=3,
			storage_cost_per_product=0.1)

		hyperparameter_config = HyperparameterConfig.objects.create(sim_market=sim_market_config,
			rl=rl_config)

		final_config = Config.objects.create(environment=env_config,
			hyperparameter=hyperparameter_config)

		expected_dict = {
			'hyperparameters': {
				'rl': {
					'gamma': 0.99,
					'batch_size': 32,
					'replay_size': 100000,
					'learning_rate': 1e-6,
					'sync_target_frames': 1000,
					'replay_start_size': 10000,
					'epsilon_decay_last_frame': 75000,
					'epsilon_start': 1.0,
					'epsilon_final': 0.1
				},
				'sim_market': {
					'max_storage': 100,
					'episode_size': 50,
					'max_price': 10,
					'max_quality': 50,
					'number_of_customers': 20,
					'production_price': 3,
					'storage_cost_per_product': 0.1
				}
			},
			'environment': {
				'task': 'training',
				'marketplace': 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
				'agents': {
					'Rule_Based Agent': {
						'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
					}
				}
			}
		}
		print('TEST STARTS HERE')
		print(final_config.as_dict())
		print(expected_dict)
		assert expected_dict == final_config.as_dict()
		print('TEST ENDS HERE')

	def test_remove_none_values_from_dict(self):
		test_dict = {'test': 'test', 'test2': None}
		assert {'test': 'test'} == remove_none_values_from_dict(test_dict)
