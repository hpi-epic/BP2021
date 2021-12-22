# import os
# import re
# import shutil
# from importlib import reload
# from unittest.mock import mock_open, patch

# import pytest
# import torch

# import agent
# import sim_market
# import tests.utils_tests as ut_t
# import training
# import utils_rl as ut_rl


# # teardown after each test
# def teardown_module(module):
# 	print('***TEARDOWN***')
# 	for f in os.listdir('./runs'):
# 		if re.match('test_*', f):
# 			shutil.rmtree('./runs/' + f)


# test_scenarios = [
# 	(sim_market.ClassicScenario(), agent.QLearningAgent(n_observation=sim_market.ClassicScenario().observation_space.shape[0], n_actions=10, optim=torch.optim.Adam)),
# 	(sim_market.MultiCompetitorScenario(), agent.QLearningAgent(n_observation=sim_market.MultiCompetitorScenario().observation_space.shape[0], n_actions=10, optim=torch.optim.Adam)),
# 	(sim_market.CircularEconomyMonopolyScenario(), agent.QLearningCEAgent(sim_market.CircularEconomyMonopolyScenario().observation_space.shape[0], n_actions=100, optim=torch.optim.Adam)),
# 	(sim_market.CircularEconomyRebuyPriceMonopolyScenario(), agent.QLearningCERebuyAgent(sim_market.CircularEconomyRebuyPriceMonopolyScenario().observation_space.shape[0], n_actions=100, optim=torch.optim.Adam)),
# 	(sim_market.CircularEconomyRebuyPriceOneCompetitor(), agent.QLearningCERebuyAgent(sim_market.CircularEconomyRebuyPriceOneCompetitor().observation_space.shape[0], n_actions=100, optim=torch.optim.Adam))]


# @pytest.mark.parametrize('environment, agent', test_scenarios)
# def test_market_scenario(environment, agent):
# 	json = ut_t.create_mock_json_rl(replay_start_size='500', sync_target_frames='100')
# 	with patch('builtins.open', mock_open(read_data=json)) as mock_file:
# 		ut_t.check_mock_file_rl(mock_file, json)
# 		# Include utils_rl again to make sure the file is read again
# 		reload(ut_rl)
# 		training.train_QLearning_agent(agent, environment, int(ut_rl.REPLAY_START_SIZE * 1.2), log_dir_prepend='test_')
