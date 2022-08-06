from unittest.mock import patch

import recommerce.rl.training_scenario as training_scenario


def test_train_q_learning_classic_scenario():
	with patch('recommerce.rl.q_learning.q_learning_training.QLearningTrainer.train_agent') as mock_train_agent:
		training_scenario.train_q_learning_classic_scenario()
		assert mock_train_agent.called


def test_train_q_learning_circular_economy_rebuy():
	with patch('recommerce.rl.q_learning.q_learning_training.QLearningTrainer.train_agent') as mock_train_agent:
		training_scenario.train_q_learning_circular_economy_rebuy()
		assert mock_train_agent.called


def test_train_continuous_a2c_circular_economy_rebuy():
	with patch('recommerce.rl.actorcritic.actorcritic_training.ActorCriticTrainer.train_agent') as mock_train_agent:
		training_scenario.train_continuous_a2c_circular_economy_rebuy()
		assert mock_train_agent.called


def test_train_stable_baselines_ppo():
	with patch('recommerce.rl.stable_baselines.sb_ppo.StableBaselinesPPO.train_agent') as mock_train_agent:
		training_scenario.train_stable_baselines_ppo()
		assert mock_train_agent.called


def test_train_stable_baselines_sac():
	with patch('recommerce.rl.stable_baselines.sb_sac.StableBaselinesSAC.train_agent') as mock_train_agent:
		training_scenario.train_stable_baselines_sac()
		assert mock_train_agent.called

# TODO: Implement this test with a good performance like the other ones.
# def test_train_rl_vs_rl():
# 	# training_scenario.train_rl_vs_rl()


# train_self_play is just a start of an already tested method.
