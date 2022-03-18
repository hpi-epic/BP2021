from django.test import TestCase

from ..models.config import *
from ..models.config import get_config_field_names


class ConfigTest(TestCase):
	def test_right_field_names_config(self):
		expected_field_names = ['environment', 'hyperparameter']
		actual_field_names = get_config_field_names(Config)
		assert expected_field_names == actual_field_names

	def test_right_field_names_environment_config(self):
		expected_field_names = ['agent', 'enable_live_draw', 'episodes', 'plot_interval', 'marketplace', 'task']
		actual_field_names = get_config_field_names(EnvironmentConfig)
		assert expected_field_names == actual_field_names

	def test_right_field_names_agents_config(self):
		expected_field_names = ['name', 'agent_class', 'modelfile']
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
