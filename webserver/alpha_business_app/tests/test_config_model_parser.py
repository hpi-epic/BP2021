import copy

from django.test import TestCase

from ..config_parser import ConfigModelParser
from ..models.agents_config import AgentsConfig
from ..models.config import Config
from ..models.environment_config import EnvironmentConfig
from ..models.hyperparameter_config import HyperparameterConfig
from ..models.rl_config import RlConfig
from ..models.sim_market_config import SimMarketConfig
from .constant_tests import EXAMPLE_HIERARCHY_DICT, EXAMPLE_RL_DICT


class ConfigModelParserTest(TestCase):
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
			'separate_markets': False,
			'marketplace': 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopoly',
			'agents': [
				{
					'name': 'Rule_Based Agent',
					'agent_class': 'recommerce.market.circular.circular_vendors.RuleBasedCERebuyAgent',
					'argument': ''
				}
			]
		}
	}

	def setUp(self) -> None:
		self.parser = ConfigModelParser()

	def test_parsing_config_dict(self):
		test_dict = copy.deepcopy(EXAMPLE_HIERARCHY_DICT)

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
		assert environment_config.separate_markets is False
		assert environment_config.episodes is None
		assert environment_config.plot_interval is None
		assert 'recommerce.market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopoly' == environment_config.marketplace
		assert environment_config.agents is not None

		environment_agents: AgentsConfig = environment_config.agents

		all_agents = environment_agents.agentconfig_set.all()
		assert 1 == len(all_agents)
		assert 'recommerce.rl.q_learning.q_learning_agent.QLearningAgent' == all_agents[0].agent_class
		assert 'QLearning Agent' == all_agents[0].name
		assert '' == all_agents[0].argument

	def test_parsing_agents(self):
		test_dict = [
			{
				'name': 'test_agent1',
				'agent_class': 'test_class',
				'argument': ''
			},
			{
				'name': 'test_agent2',
				'agent_class': 'test_class',
				'argument': '1234'
			}
		]
		agents = self.parser._parse_agents_to_datastructure(test_dict)
		all_agents = agents.agentconfig_set.all()

		assert 'test_agent1' == all_agents[0].name
		assert 'test_class' == all_agents[0].agent_class
		assert '' == all_agents[0].argument

		assert 'test_agent2' == all_agents[1].name
		assert 'test_class' == all_agents[1].agent_class
		assert '1234' == all_agents[1].argument

	def test_parse_rl(self):
		test_dict = EXAMPLE_RL_DICT.copy()

		final_config = self.parser.parse_config_dict_to_datastructure('hyperparameter', test_dict)

		assert HyperparameterConfig == type(final_config)

		# assert all hyperparameters
		hyperparameter_rl_config: RlConfig = final_config.rl
		hyperparameter_sim_market_config: SimMarketConfig = final_config.sim_market
		assert hyperparameter_rl_config is not None
		assert hyperparameter_sim_market_config is None

		assert 0.99 == hyperparameter_rl_config.gamma
		assert 32 == hyperparameter_rl_config.batch_size
		assert 100000 == hyperparameter_rl_config.replay_size
		assert 1e-6 == hyperparameter_rl_config.learning_rate
		assert 1000 == hyperparameter_rl_config.sync_target_frames
		assert 10000 == hyperparameter_rl_config.replay_start_size
		assert 75000 == hyperparameter_rl_config.epsilon_decay_last_frame
		assert 1.0 == hyperparameter_rl_config.epsilon_start
		assert 0.1 == hyperparameter_rl_config.epsilon_final
		assert hyperparameter_rl_config.n_steps is None
		assert hyperparameter_rl_config.n_epochs is None
		assert hyperparameter_rl_config.tau is None
		assert hyperparameter_rl_config.clip_range is None
		assert hyperparameter_rl_config.neurones_per_hidden_layer is None
		assert hyperparameter_rl_config.ent_coef is None
		assert hyperparameter_rl_config.buffer_size is None
		assert hyperparameter_rl_config.learning_starts is None
