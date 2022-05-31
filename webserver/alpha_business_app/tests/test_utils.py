from django.test import TestCase

from ..utils import get_structure_dict_for, to_config_keyword


class UtilsTest(TestCase):
	def test_get_structure_dict_for(self):
		expected_dict = {
			'environment': {
				'task': None,
				'enable_live_draw': None,
				'episodes': None,
				'plot_interval': None,
				'marketplace': None,
				'agents': []
			},
			'hyperparameter': {
				'sim_market': {
					'class': None,
					'max_storage': None,
					'episode_length': None,
					'max_price': None,
					'max_quality': None,
					'number_of_customers': None,
					'production_price': None,
					'storage_cost_per_product': None
				},
				'rl': {
					'replay_size': None,
					'epsilon_start': None,
					'replay_start_size': None,
					'epsilon_decay_last_frame': None,
					'testvalue2': None,
					'sync_target_frames': None,
					'batch_size': None,
					'threshold': None,
					'epsilon_final': None,
					'stable_baseline_test': None,
					'gamma': None,
					'learning_rate': None
				}
			}
		}
		assert expected_dict == get_structure_dict_for('')

	def test_get_structure_dict_for_rl(self):
		expected_dict = {
			'replay_size': None,
			'epsilon_start': None,
			'replay_start_size': None,
			'epsilon_decay_last_frame': None,
			'testvalue2': None,
			'sync_target_frames': None,
			'batch_size': None,
			'threshold': None,
			'epsilon_final': None,
			'stable_baseline_test': None,
			'gamma': None,
			'learning_rate': None
		}
		assert expected_dict == get_structure_dict_for('rl')

	def test_get_structure_dict_for_sim_market(self):
		expected_dict = {
			'class': None,
			'max_storage': None,
			'episode_length': None,
			'max_price': None,
			'max_quality': None,
			'number_of_customers': None,
			'production_price': None,
			'storage_cost_per_product': None
		}
		assert expected_dict == get_structure_dict_for('sim_market')

	def test_get_structure_dict_for_environment(self):
		expected_dict = {
			'task': None,
			'enable_live_draw': None,
			'episodes': None,
			'plot_interval': None,
			'marketplace': None,
			'agents': []
		}
		assert expected_dict == get_structure_dict_for('environment')

	def test_get_structure_dict_for_agents(self):
		assert [] == get_structure_dict_for('agents')

	def test_to_config_keyword(self):
		from ..models.config import Config
		assert '' == to_config_keyword(Config)
		from ..models.agents_config import AgentsConfig
		assert 'agents' == to_config_keyword(AgentsConfig)
		from ..models.environment_config import EnvironmentConfig
		assert 'environment' == to_config_keyword(EnvironmentConfig)
		from ..models.hyperparameter_config import HyperparameterConfig
		assert 'hyperparameter' == to_config_keyword(HyperparameterConfig)
		from ..models.rl_config import RlConfig
		assert 'rl' == to_config_keyword(RlConfig)
		from ..models.sim_market_config import SimMarketConfig
		assert 'sim_market' == to_config_keyword(SimMarketConfig)
