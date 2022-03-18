from django.test import TestCase

from ..models.config import *
from ..models.config import get_config_field_names


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
		print('HERE', to_config_class_name('CE Rebuy Agent (QLearning)'))
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
