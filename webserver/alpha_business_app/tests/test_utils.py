from django.test import TestCase

from ..utils import (convert_python_type_to_django_type, get_all_possible_rl_hyperparameter, get_all_possible_sim_market_hyperparameter,
                     get_structure_dict_for, to_config_keyword)


class UtilsTest(TestCase):
    def test_get_structure_dict_for_config(self):
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
            'epsilon_final': None,
            'stable_baseline_test': None,
            'gamma': None,
            'learning_rate': None
        }
        assert expected_dict == get_structure_dict_for('rl')

    def test_get_structure_dict_for_sim_market(self):
        expected_dict = {
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

    def test_get_all_rl_parameter(self):
        expected_parameter = {
            ('gamma', float),
            ('batch_size', int),
            ('replay_start_size', int),
            ('sync_target_frames', int),
            ('epsilon_decay_last_frame', int),
            ('replay_size', int),
            ('epsilon_final', float),
            ('stable_baseline_test', float),
            ('testvalue2', float),
            ('epsilon_start', float),
            ('learning_rate', float)
        }
        assert expected_parameter == get_all_possible_rl_hyperparameter()

    def test_get_all_sim_market_parameter(self):
        expected_parameter = {
            ('max_price', int),
            ('production_price', int),
            ('episode_length', int),
            ('max_quality', int),
            ('max_storage', int),
            ('storage_cost_per_product', (int, float)),
            ('number_of_customers', int)
        }
        assert expected_parameter == get_all_possible_sim_market_hyperparameter()

    def test_convert_to_django_type(self):
        assert "<class 'django.db.models.fields.IntegerField'>" == convert_python_type_to_django_type(int)
        assert "<class 'django.db.models.fields.FloatField'>" == convert_python_type_to_django_type(float)
        assert "<class 'django.db.models.fields.CharField'>" == convert_python_type_to_django_type(str)
        assert "<class 'django.db.models.fields.FloatField'>" == convert_python_type_to_django_type((int, float))
