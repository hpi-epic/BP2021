EXAMPLE_POST_REQUEST_ARGUMENTS = {
		'csrfmiddlewaretoken': ['PHZ3VkxiJkrk2gnBCkgNfYJAdUsdb4V5e7CO26nJuENMtSas7BVapRGJJ0B3t9HZ'],
		'action': ['start'],
		'experiment_name': ['test_experiment'],
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

EXAMPLE_HIERARCHIE_DICT = {
		'environment': {
			'task': 'training',
			'marketplace': 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
			'enable_live_draw': False,
			'agents': {
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		},
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
		}
	}

EXAMPLE_HIERARCHIE_DICT2 = {
		'environment': {
			'task': 'training',
			'marketplace': 'market.circular.circular_sim_market.CircularEconomyRebuyPriceMonopolyScenario',
			'enable_live_draw': False,
			'agents': {
				'Rule_Based Agent': {
					'agent_class': 'agents.vendors.RuleBasedCERebuyAgent'
				}
			}
		},
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
		}
	}