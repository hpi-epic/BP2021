import json
import os
from unittest import mock
from unittest.mock import patch

import recommerce.rl.training_scenario as training_scenario
from recommerce.configuration.path_manager import PathManager

# we want to edit the market_config in this case here
with open(os.path.join(PathManager.user_path, "configuration_files", "market_config.json"), 'r') as f:
	market_config_json = json.load(f)
	market_config_json["support_continuous_action_space"] = True

builtin_open = open  # save the unpatched version


def mock_open(*args, **kwargs):
	print(args[0])
	if args[0] == os.path.join(PathManager.user_path, "configuration_files", "market_config.json"):
		return mock.mock_open(read_data=json.dumps(market_config_json))(*args, **kwargs)
	return builtin_open(*args, **kwargs)


def test_train_q_learning_classic_scenario():
	with patch('recommerce.rl.q_learning.q_learning_training.QLearningTrainer.train_agent') as mock_train_agent:
		training_scenario.train_q_learning_classic_scenario()
		assert mock_train_agent.called


def test_train_q_learning_circular_economy_rebuy():
	with patch('recommerce.rl.q_learning.q_learning_training.QLearningTrainer.train_agent') as mock_train_agent:
		training_scenario.train_q_learning_circular_economy_rebuy()
		assert mock_train_agent.called


def test_train_continuous_a2c_circular_economy_rebuy():
	with mock.patch("builtins.open", mock_open):
		with patch('recommerce.rl.actorcritic.actorcritic_training.ActorCriticTrainer.train_agent') as mock_train_agent:
			training_scenario.train_continuous_a2c_circular_economy_rebuy()
			assert mock_train_agent.called


def test_train_stable_baselines_ppo():
	with mock.patch("builtins.open", mock_open):
		with patch('recommerce.rl.stable_baselines.sb_ppo.StableBaselinesPPO.train_agent') as mock_train_agent:
			training_scenario.train_stable_baselines_ppo()
			assert mock_train_agent.called



def test_train_stable_baselines_sac():
	with mock.patch("builtins.open", mock_open):
		with patch('recommerce.rl.stable_baselines.sb_sac.StableBaselinesSAC.train_agent') as mock_train_agent:
			training_scenario.train_stable_baselines_sac()
			assert mock_train_agent.called

# TODO: Implement this test with a good performance like the other ones.
# def test_train_rl_vs_rl():
# 	# training_scenario.train_rl_vs_rl()


# train_self_play is just a start of an already tested method.
