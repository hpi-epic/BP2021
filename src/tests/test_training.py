# import os
# import re
# import shutil
# from importlib import reload
# from unittest.mock import mock_open, patch

# import pytest
# import torch

# import agents.vendors as vendors
# import configuration.config as config
# import market.sim_market as sim_market
# import rl.training as training
# import tests.utils_tests as ut_t

# test_scenarios = [
# 	(sim_market.ClassicScenario, vendors.QLearningAgent, 10),
# 	(sim_market.MultiCompetitorScenario, vendors.QLearningAgent, 10),
# 	(sim_market.CircularEconomyMonopolyScenario, vendors.QLearningCEAgent, 100),
# 	(sim_market.CircularEconomyRebuyPriceMonopolyScenario, vendors.QLearningCERebuyAgent, 100),
# 	(sim_market.CircularEconomyRebuyPriceOneCompetitor, vendors.QLearningCERebuyAgent, 100)
# ]


# @pytest.mark.parametrize('environment, agent, n_actions', test_scenarios)
# def test_market_scenario(environment, agent, n_actions):
# 	json = ut_t.create_mock_json(rl=ut_t.create_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
# 	with patch('builtins.open', mock_open(read_data=json)) as mock_file, \
# 		patch('rl.training.SummaryWriter'):
# 		ut_t.check_mock_file(mock_file, json)
# 		# Include utils_rl again to make sure the file is read again
# 		reload(config)
# 		environment = environment()
# 		agent = agent(environment.observation_space.shape[0], n_actions=n_actions, optim=torch.optim.Adam)
# 		training.train_QLearning_agent(agent, environment, int(config.REPLAY_START_SIZE * 1.2))


# def test_training_with_tensorboard():
# 	json = ut_t.create_mock_json(rl=ut_t.create_mock_json_rl(replay_start_size='500', sync_target_frames='100'))
# 	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
# 		ut_t.check_mock_file(mock_file, json)
# 		# Include utils_rl again to make sure the file is read again
# 		reload(config)
# 		environment = sim_market.ClassicScenario()
# 		agent = vendors.QLearningAgent(environment.observation_space.shape[0], n_actions=10, optim=torch.optim.Adam)
# 		training.train_QLearning_agent(agent, environment, int(config.REPLAY_START_SIZE * 1.2), log_dir_prepend='test_')
# 	for f in os.listdir('./runs'):
# 		if re.match('test_*', f):
# 			shutil.rmtree('./runs/' + f)
# 	if os.listdir('./runs') == []:
# 		os.rmdir('./runs')
