from django.test import TestCase

from ..configuration_parser import ConfigurationParser


class ConfigTest(TestCase):
	expected_dict = {
		'hyperparameter': {
			'rl': {
				'gamma': '0.99',
				'batch_size': '32',
				'replay_size': '100000',
				'learning_rate': '1e-06',
				'sync_target_frames': '1000',
				'replay_start_size': '10000',
				'epsilon_decay_last_frame': '75000',
				'epsilon_start': '1.0',
				'epsilon_final': '0.1'
			},
			'sim_market': {
				'max_storage': '100',
				'episode_length': '50',
				'max_price': '10',
				'max_quality': '50',
				'number_of_customers': '20',
				'production_price': '3',
				'storage_cost_per_product': '0.1'
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

	def setUp(self) -> None:
		self.parser = ConfigurationParser()

	def test_parsing_flat_dict(self):
		test_dict = {
			'csrfmiddlewaretoken': ['PHZ3VkxiJkrk2gnBCkgNfYJAdUsdb4V5e7CO26nJuENMtSas7BVapRGJJ0B3t9HZ'],
			'action': ['start'],
			'environment-task': ['training'],
			'environment-episodes': [''],
			'environment-plot_interval': [''],
			'environment-marketplace': ['market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'],
			'environment-agents-name': ['Rule_Based Agent'],
			'environment-agents-agent_class': ['agents.vendors.RuleBasedCERebuyAgent'],
			'environment-agents-argument': [''],
			'hyperparameter-rl-gamma': ['0.99'],
			'hyperparameter-rl-batch_size': ['32'],
			'hyperparameter-rl-replay_size': ['100000'],
			'hyperparameter-rl-learning_rate': ['1e-06'],
			'hyperparameter-rl-sync_target_frames': ['1000'],
			'hyperparameter-rl-replay_start_size': ['10000'],
			'hyperparameter-rl-epsilon_decay_last_frame': ['75000'],
			'hyperparameter-rl-epsilon_start': ['1.0'],
			'hyperparameter-rl-epsilon_final': ['0.1'],
			'hyperparameter-sim_market-max_storage': ['100'],
			'hyperparameter-sim_market-episode_length': ['50'],
			'hyperparameter-sim_market-max_price': ['10'],
			'hyperparameter-sim_market-max_quality': ['50'],
			'hyperparameter-sim_market-number_of_customers': ['20'],
			'hyperparameter-sim_market-production_price': ['3'],
			'hyperparameter-sim_market-storage_cost_per_product': ['0.1']
		}
		assert self.expected_dict == self.parser.flat_dict_to_hierarchical_config_dict(test_dict)

	def test_remove_keyword_parts(self):
		test_dict = {
			'environment-task': ['training'],
			'environment-episodes': [''],
			'environment-plot_interval': ['']
		}
		expected_dict = {
			'task': ['training'],
			'episodes': [''],
			'plot_interval': ['']
		}

		assert expected_dict == self.parser._get_items_key_starts_with(test_dict, 'environment-')

	def test_substract_dicts(self):
		test_dict1 = {
			'a': '123',
			'b': '456',
			'test': '789'
		}
		test_dict2 = {
			'a': '123',
			'b': '456'
		}
		expected_dict = {
			'test': '789'
		}
		assert expected_dict == self.parser._substract_dicts(test_dict1, test_dict2)

	def test_flat_environment(self):
		test_dict = {
			'task': ['training'],
			'episodes': [''],
			'plot_interval': [''],
			'marketplace': ['market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'],
			'agents-name': ['Rule_Based Agent'],
			'agents-agent_class': ['agents.vendors.RuleBasedCERebuyAgent'],
			'agents-argument': [''],
		}
		assert self.expected_dict['environment'] == self.parser._flat_environment_to_hierarchical(test_dict)

	def test_flat_agents(self):
		test_dict = {
			'name': ['Rule_Based Agent'],
			'agent_class': ['agents.vendors.RuleBasedCERebuyAgent'],
			'argument': [''],
		}
		assert self.expected_dict['environment']['agents'] == self.parser._flat_agents_to_hierarchical(test_dict)

	def test_flat_hyperparameters(self):
		test_dict = {
			'rl-gamma': ['0.99'],
			'rl-batch_size': ['32'],
			'rl-replay_size': ['100000'],
			'rl-learning_rate': ['1e-06'],
			'rl-sync_target_frames': ['1000'],
			'rl-replay_start_size': ['10000'],
			'rl-epsilon_decay_last_frame': ['75000'],
			'rl-epsilon_start': ['1.0'],
			'rl-epsilon_final': ['0.1'],
			'sim_market-max_storage': ['100'],
			'sim_market-episode_length': ['50'],
			'sim_market-max_price': ['10'],
			'sim_market-max_quality': ['50'],
			'sim_market-number_of_customers': ['20'],
			'sim_market-production_price': ['3'],
			'sim_market-storage_cost_per_product': ['0.1']
		}
		assert self.expected_dict['hyperparameter'] == self.parser._flat_hyperparameter_to_hierarchical(test_dict)
