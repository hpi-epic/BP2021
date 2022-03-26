import copy

from django.contrib.sessions.middleware import SessionMiddleware
from django.test import TestCase
from django.test.client import RequestFactory

# from ..buttons import ButtonHandler
from ..config_merger import ConfigMerger
from ..config_parser import ConfigModelParser
from ..models.config import Config, EnvironmentConfig, HyperparameterConfig, RlConfig
from .constant_tests import EMPTY_STRUCTURE_CONFIG, EXAMPLE_HIERARCHIE_DICT, EXAMPLE_HIERARCHIE_DICT2


class ConfigMergerTest(TestCase):

	# def test_no_configs_selected(self):
	# 	print('TESTSTART')
	# 	request = self._setup_request({})
	# 	button_handler = ButtonHandler(request, view='configurator.html', rendering_method='config_files')

	# 	button_handler.do_button_click()
	# 	with patch('alpha_business_app.buttons.ButtonHandler._render_files') as render_mock:
	# 		render_mock.assert_called_once()
	# 		render_mock.assert_called_once_with(request, 'configurator.html', {'all_configurations': Config.objects.all()})
	# 	print('TESTEND')

	def test_merge_one_config(self):
		test_dict = copy.deepcopy(EXAMPLE_HIERARCHIE_DICT)
		config_object = ConfigModelParser().parse_config(test_dict)
		config_dict = config_object.as_dict()
		expected_dict = copy.deepcopy(config_dict)
		expected_dict['environment']['episodes'] = None
		expected_dict['environment']['plot_interval'] = None
		empty_config = Config.get_empty_structure_dict()
		merger = ConfigMerger()
		actual_config = merger._merge_config_into_base_config(empty_config, config_dict)

		assert expected_dict == actual_config

	def test_merge_two_configs_without_conflicts(self):
		test_environment_config = EnvironmentConfig.objects.create(task='training')
		test_config1 = Config.objects.create(environment=test_environment_config)

		test_rl_config = RlConfig.objects.create(gamma=0.99)
		test_hyper_parameter_config = HyperparameterConfig.objects.create(rl=test_rl_config)
		test_config2 = Config.objects.create(hyperparameter=test_hyper_parameter_config)

		merger = ConfigMerger()
		final_dict, error_dict = merger.merge_config_objects([test_config1.id, test_config2.id])

		expected_dict = copy.deepcopy(EMPTY_STRUCTURE_CONFIG)
		expected_dict['hyperparameter']['rl']['gamma'] = 0.99
		expected_dict['environment']['task'] = 'training'

		assert expected_dict == final_dict
		assert EMPTY_STRUCTURE_CONFIG == error_dict

	def test_merge_two_small_configs_with_conflicts(self):
		test_environment_config1 = EnvironmentConfig.objects.create(task='training')
		test_config1 = Config.objects.create(environment=test_environment_config1)

		test_environment_config2 = EnvironmentConfig.objects.create(task='monitoring')
		test_config2 = Config.objects.create(environment=test_environment_config2)

		merger = ConfigMerger()
		final_dict, error_dict = merger.merge_config_objects([test_config1.id, test_config2.id])

		expected_dict = copy.deepcopy(EMPTY_STRUCTURE_CONFIG)
		expected_dict['environment']['task'] = 'monitoring'

		expected_error = copy.deepcopy(EMPTY_STRUCTURE_CONFIG)
		expected_error['environment']['task'] = 'changed environment task from training to monitoring'

		assert expected_dict == final_dict
		assert expected_error == error_dict

	def test_merge_two_configs_with_conflicts(self):
		test_config1 = copy.deepcopy(EXAMPLE_HIERARCHIE_DICT)
		test_config2 = copy.deepcopy(EXAMPLE_HIERARCHIE_DICT2)

		parser = ConfigModelParser()
		config_object1 = parser.parse_config(test_config1)
		config_object2 = parser.parse_config(test_config2)

		merger = ConfigMerger()
		final_config, error_dict = merger.merge_config_objects([config_object1.id, config_object2.id])

		expected_final_config = {
			'environment': {
				'enable_live_draw': True,
				'episodes': None,
				'plot_interval': None,
				'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
				'task': 'monitoring',
				'agents': {
					'Rule_Based Agent': {
						'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent'
					},
					'CE Rebuy Agent (QLearning)': {
						'agent_class': 'recommerce.rl.q_learning.q_learning_agent.QLearningCERebuyAgent',
						'argument': 'CircularEconomyRebuyPriceMonopolyScenario_QLearningCERebuyAgent.dat'
					}
				}
			}, 'hyperparameter': {
				'rl': {
					'gamma': 0.8,
					'batch_size': 16,
					'replay_size': 10000,
					'learning_rate': 1e-05,
					'sync_target_frames': 100,
					'replay_start_size': 1000,
					'epsilon_decay_last_frame': 7500,
					'epsilon_start': 0.9, 'epsilon_final': 0.2
				},
				'sim_market': {
					'max_storage': 80,
					'episode_length': 80,
					'max_price': 90, 'max_quality': 50,
					'number_of_customers': 6,
					'production_price': 1,
					'storage_cost_per_product': 0.7
				}
			}
		}
		expected_error_dict = {
			'environment': {
				'enable_live_draw': 'changed environment enable_live_draw from False to True',
				'episodes': None,
				'plot_interval': None,
				'marketplace': 'changed environment marketplace from market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario to recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
				'task': 'changed environment task from training to monitoring',
				'agents': 'multiple Rule_Based Agent'
			},
			'hyperparameter': {
				'rl': {
					'gamma': 'changed hyperparameter-rl gamma from 0.99 to 0.8',
					'batch_size': 'changed hyperparameter-rl batch_size from 32 to 16',
					'replay_size': 'changed hyperparameter-rl replay_size from 100000 to 10000',
					'learning_rate': 'changed hyperparameter-rl learning_rate from 1e-06 to 1e-05',
					'sync_target_frames': 'changed hyperparameter-rl sync_target_frames from 1000 to 100',
					'replay_start_size': 'changed hyperparameter-rl replay_start_size from 10000 to 1000',
					'epsilon_decay_last_frame': 'changed hyperparameter-rl epsilon_decay_last_frame from 75000 to 7500',
					'epsilon_start': 'changed hyperparameter-rl epsilon_start from 1.0 to 0.9',
					'epsilon_final': 'changed hyperparameter-rl epsilon_final from 0.1 to 0.2'
				},
				'sim_market': {
					'max_storage': 'changed hyperparameter-sim_market max_storage from 100 to 80',
					'episode_length': 'changed hyperparameter-sim_market episode_length from 50 to 80', 'max_price':
					'changed hyperparameter-sim_market max_price from 10 to 90',
					'max_quality': None,
					'number_of_customers': 'changed hyperparameter-sim_market number_of_customers from 20 to 6',
					'production_price': 'changed hyperparameter-sim_market production_price from 3 to 1',
					'storage_cost_per_product':
					'changed hyperparameter-sim_market storage_cost_per_product from 0.1 to 0.7'
				}
			}
		}
		assert expected_final_config == final_config
		assert expected_error_dict == error_dict

	def test_merge_one_agent(self):
		test_agent_dict = {
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		merger = ConfigMerger()
		actual = merger._merge_agents_into_base_agents({}, test_agent_dict)
		assert test_agent_dict == actual

	def test_merge_two_same_agents(self):
		test_agent_dict1 = {
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		test_agent_dict2 = {
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		merger = ConfigMerger()
		actual = merger._merge_agents_into_base_agents(test_agent_dict1, test_agent_dict2)

		expected_dict = {
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		expected_error = copy.deepcopy(EMPTY_STRUCTURE_CONFIG)
		expected_error['environment']['agents'] = 'multiple Rule_Based Agent'

		assert expected_dict == actual
		assert expected_error == merger.error_dict

	def test_merge_two_agents(self):
		test_agent_dict = {
				'test agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				},
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		merger = ConfigMerger()
		actual = merger._merge_agents_into_base_agents({}, test_agent_dict)
		expected_dict = {
				'test agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				},
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		assert expected_dict == actual

	def test_update_error_dict(self):
		expected_error_dict = copy.deepcopy(EMPTY_STRUCTURE_CONFIG)
		expected_error_dict['environment']['task'] = 'test_error'

		merger = ConfigMerger()
		merger._update_error_dict(['environment', 'task'], 'test_error')

		assert expected_error_dict == merger.error_dict

	def test_update_error_dict2(self):
		expected_error_dict = copy.deepcopy(EMPTY_STRUCTURE_CONFIG)
		expected_error_dict['environment']['agents'] = 'test_error'

		merger = ConfigMerger()
		merger._update_error_dict(['environment', 'agents'], 'test_error')

		assert expected_error_dict == merger.error_dict

	def test_update_error_dict3(self):
		expected_error_dict = copy.deepcopy(EMPTY_STRUCTURE_CONFIG)
		expected_error_dict['hyperparameter']['rl']['gamma'] = 'test_error'

		merger = ConfigMerger()
		merger._update_error_dict(['hyperparameter', 'rl', 'gamma'], 'test_error')

		assert expected_error_dict == merger.error_dict

	def _setup_request(self, arguments: dict) -> RequestFactory:
		request = RequestFactory().post('configurator.html', {'action': 'pre-fill', **arguments})
		middleware = SessionMiddleware(request)
		middleware.process_request(request)
		request.session.save()
		return request
