from django.test import TestCase

from ..utils import *
from .constant_tests import EMPTY_STRUCTURE_CONFIG


class UtilsTest(TestCase):
	def test_get_structure_dict_for_config(self):
		expected_dict = {
			'environment': {
				'task': None,
				'separate_markets': None,
				'episodes': None,
				'plot_interval': None,
				'marketplace': None,
				'agents': []
			},
			'hyperparameter': {
				'sim_market': {
					'common_state_visibility': None,
					'opposite_own_state_visibility': None,
					'max_storage': None,
					'production_price': None,
					'max_quality': None,
					'episode_length': None,
					'storage_cost_per_product': None,
					'number_of_customers': None,
					'reward_mixed_profit_and_difference': None,
					'max_price': None
				},
				'rl': {
					'n_steps': None,
					'n_epochs': None,
					'sync_target_frames': None,
					'tau': None,
					'replay_size': None,
					'clip_range': None,
					'gamma': None,
					'neurones_per_hidden_layer': None,
					'replay_start_size': None,
					'learning_starts': None,
					'ent_coef': None,
					'epsilon_final': None,
					'epsilon_decay_last_frame': None,
					'epsilon_start': None,
					'batch_size': None,
					'learning_rate': None,
					'buffer_size': None
				}
			}
		}
		assert expected_dict == get_structure_dict_for('')

	def test_get_structure_dict_for_rl(self):
		expected_dict = EMPTY_STRUCTURE_CONFIG.copy()['hyperparameter']['rl']
		assert expected_dict == get_structure_dict_for('rl')

	def test_get_structure_dict_for_sim_market(self):
		expected_dict = EMPTY_STRUCTURE_CONFIG.copy()['hyperparameter']['sim_market']
		assert expected_dict == get_structure_dict_for('sim_market')

	def test_get_structure_dict_for_environment(self):
		expected_dict = EMPTY_STRUCTURE_CONFIG.copy()['environment']
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

	def test_get_all_rl_parameter(self):
		expected_parameter = {
			('replay_start_size', int),
			('n_epochs', int),
			('tau', float),
			('replay_size', int),
			('n_steps', int),
			('buffer_size', int),
			('epsilon_decay_last_frame', int),
			('epsilon_final', float),
			('clip_range', float),
			('ent_coef', (str, float)),
			('learning_starts', int),
			('epsilon_start', float),
			('gamma', float),
			('learning_rate', float),
			('neurones_per_hidden_layer', int),
			('sync_target_frames', int),
			('batch_size', int)
		}
		assert expected_parameter == get_all_possible_rl_hyperparameter()

	def test_get_all_sim_market_parameter(self):
		expected_parameter = {
			('max_price', int),
			('episode_length', int),
			('max_storage', int),
			('number_of_customers', int),
			('opposite_own_state_visibility', bool),
			('storage_cost_per_product', (int, float)),
			('max_quality', int),
			('production_price', int),
			('common_state_visibility', bool),
			('reward_mixed_profit_and_difference', bool)
		}
		assert expected_parameter == get_all_possible_sim_market_hyperparameter()

	def test_convert_to_django_type(self):
		assert "<class 'django.db.models.fields.IntegerField'>" == convert_python_type_to_django_type(int)
		assert "<class 'django.db.models.fields.FloatField'>" == convert_python_type_to_django_type(float)
		assert "<class 'django.db.models.fields.CharField'>" == convert_python_type_to_django_type(str)
		assert "<class 'django.db.models.fields.FloatField'>" == convert_python_type_to_django_type((int, float))
		assert "<class 'django.db.models.fields.BooleanField'>" == convert_python_type_to_django_type(bool)
