from django.test import TestCase

from ..config_parser import ConfigFlatDictParser, ConfigModelParser
from ..models.config import AgentsConfig, Config, EnvironmentConfig, RlConfig, SimMarketConfig
from .constant_tests import EXAMPLE_HIERARCHY_DICT


class ConfigParserTest(TestCase):
	expected_dict = {
		'hyperparameter': {
			'rl': {
				'gamma': 0.99,
				'batch_size': 32,
				'replay_size': 100000,
				'learning_rate': 1e-06,
				'sync_target_frames': 1000,
				'replay_start_size': 10000,
				'epsilon_decay_last_frame': 75000,
				'epsilon_start': 1.0,
				'epsilon_final': 0.1
			},
			'sim_market': {
				'max_storage': 100,
				'episode_length': 50,
				'max_price': 10,
				'max_quality': 50,
				'number_of_customers': 20,
				'production_price': 3,
				'storage_cost_per_product': 0.1
			}
		},
		'environment': {
			'task': 'training',
			'enable_live_draw': False,
			'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
			'agents': {
				'Rule_Based Agent': {
					'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
					'argument': ''
				}
			}
		}
	}

	def setUp(self) -> None:
		self.flat_parser = ConfigFlatDictParser()
		self.parser = ConfigModelParser()

	def test_parsing_flat_dict(self):
		test_dict = {
			'csrfmiddlewaretoken': ['PHZ3VkxiJkrk2gnBCkgNfYJAdUsdb4V5e7CO26nJuENMtSas7BVapRGJJ0B3t9HZ'],
			'action': ['start'],
			'environment-task': ['training'],
			'environment-episodes': [''],
			'environment-plot_interval': [''],
			'environment-marketplace': ['recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'],
			'environment-agents-name': ['Rule_Based Agent'],
			'environment-agents-agent_class': ['recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent'],
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

		assert self.expected_dict == self.flat_parser.flat_dict_to_hierarchical_config_dict(test_dict)

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

		assert expected_dict == self.flat_parser._get_items_key_starts_with(test_dict, 'environment-')

	def test_substract_dicts(self):
		test_dict1 = {
			'a': 123,
			'b': 456,
			'test': 789
		}
		test_dict2 = {
			'a': 123,
			'b': 456
		}
		expected_dict = {
			'test': 789
		}
		assert expected_dict == self.flat_parser._substract_dicts(test_dict1, test_dict2)

	def test_flat_environment(self):
		test_dict = {
			'task': ['training'],
			'episodes': [''],
			'plot_interval': [''],
			'enable_live_draw': [''],
			'marketplace': ['recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario'],
			'agents-name': ['Rule_Based Agent'],
			'agents-agent_class': ['recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent'],
			'agents-argument': [''],
		}
		expected_environment_dict = EXAMPLE_HIERARCHY_DICT['environment'].copy()
		expected_environment_dict['enable_live_draw'] = True
		assert expected_environment_dict == self.flat_parser._flat_environment_to_hierarchical(test_dict)

	def test_flat_agents(self):
		test_dict = {
			'name': ['Rule_Based Agent'],
			'agent_class': ['recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent'],
			'argument': [''],
		}
		assert EXAMPLE_HIERARCHY_DICT['environment']['agents'] == self.flat_parser._flat_agents_to_hierarchical(test_dict)

	def test_flat_hyperparameters(self):
		test_dict = {
			'rl-gamma': [0.99],
			'rl-batch_size': [32],
			'rl-replay_size': [100000],
			'rl-learning_rate': [1e-06],
			'rl-sync_target_frames': [1000],
			'rl-replay_start_size': [10000],
			'rl-epsilon_decay_last_frame': [75000],
			'rl-epsilon_start': [1.0],
			'rl-epsilon_final': [0.1],
			'sim_market-max_storage': [100],
			'sim_market-episode_length': [50],
			'sim_market-max_price': [10],
			'sim_market-max_quality': [50],
			'sim_market-number_of_customers': [20],
			'sim_market-production_price': [3],
			'sim_market-storage_cost_per_product': [0.1]
		}
		assert EXAMPLE_HIERARCHY_DICT['hyperparameter'] == self.flat_parser._flat_hyperparameter_to_hierarchical(test_dict)

	def test_converting_to_int_or_float(self):
		assert 1 == self.flat_parser._converted_to_int_or_float_if_possible('1')
		assert 0.1 == self.flat_parser._converted_to_int_or_float_if_possible('0.1')
		assert 1e-6 == self.flat_parser._converted_to_int_or_float_if_possible('1e-6')
		assert 'string' == self.flat_parser._converted_to_int_or_float_if_possible('string')

	# parsing hierarchical
	def test_parsing_config_dict(self):
		test_dict = EXAMPLE_HIERARCHY_DICT.copy()

		final_config = self.parser.parse_config(test_dict)

		assert Config == type(final_config)
		assert final_config.hyperparameter is not None

		# assert all hyperparameters
		hyperparameter_rl_config: RlConfig = final_config.hyperparameter.rl
		hyperparameter_sim_market_config: SimMarketConfig = final_config.hyperparameter.sim_market

		assert hyperparameter_rl_config is not None
		assert final_config.hyperparameter.sim_market is not None

		assert 0.99 == hyperparameter_rl_config.gamma
		assert 32 == hyperparameter_rl_config.batch_size
		assert 100000 == hyperparameter_rl_config.replay_size
		assert 1e-06 == hyperparameter_rl_config.learning_rate
		assert 1000 == hyperparameter_rl_config.sync_target_frames
		assert 10000 == hyperparameter_rl_config.replay_start_size
		assert 75000 == hyperparameter_rl_config.epsilon_decay_last_frame
		assert 1.0 == hyperparameter_rl_config.epsilon_start
		assert 0.1 == hyperparameter_rl_config.epsilon_final

		assert 100 == hyperparameter_sim_market_config.max_storage
		assert 50 == hyperparameter_sim_market_config.episode_length
		assert 10 == hyperparameter_sim_market_config.max_price
		assert 50 == hyperparameter_sim_market_config.max_quality
		assert 20 == hyperparameter_sim_market_config.number_of_customers
		assert 3 == hyperparameter_sim_market_config.production_price
		assert 0.1 == hyperparameter_sim_market_config.storage_cost_per_product

		# assert all environment
		assert final_config.environment is not None
		environment_config: EnvironmentConfig = final_config.environment
		assert 'training' == environment_config.task
		assert environment_config.enable_live_draw is False
		assert environment_config.episodes is None
		assert environment_config.plot_interval is None
		assert 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario' == environment_config.marketplace
		assert environment_config.agents is not None

		environment_agents: AgentsConfig = environment_config.agents

		all_agents = environment_agents.agentconfig_set.all()
		assert 1 == len(all_agents)
		assert 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent' == all_agents[0].agent_class
		assert 'Rule_Based Agent' == all_agents[0].name
		assert '' == all_agents[0].argument

	def test_parsing_agents(self):
		test_dict = {
			'test_agent1': {
				'agent_class': 'test_class'
			},
			'test_agent2': {
				'agent_class': 'test_class',
				'argument': '1234'
			}
		}
		agents = self.parser._parse_agents_to_datastructure(test_dict)
		all_agents = agents.agentconfig_set.all()

		assert 'test_agent1' == all_agents[0].name
		assert 'test_class' == all_agents[0].agent_class

		assert 'test_agent2' == all_agents[1].name
		assert 'test_class' == all_agents[1].agent_class
		assert '1234' == all_agents[1].argument
