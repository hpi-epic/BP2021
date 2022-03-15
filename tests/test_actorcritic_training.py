import os
import re
import time
from unittest.mock import patch

import pytest

import alpha_business.market.circular.circular_sim_market as circular_market
import alpha_business.market.linear.linear_sim_market as linear_market
import alpha_business.rl.actorcritic_agent as actorcritic_agent
from alpha_business.rl.actorcritic_training import ActorCriticTrainer


def teardown_module(module):
	print('***TEARDOWN***')
	# we need to sleep because sometimes the subfolder is still being used when we try to remove it
	time.sleep(0.001)
	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), os.pardir, 'results', 'trainedModels')):
		if re.match('test_*', file_name):
			assert False, 'Test files were not mocked correctly'


test_scenarios = [
	(linear_market.ClassicScenario, actorcritic_agent.DiscreteACALinear, True),
	(linear_market.ClassicScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(linear_market.ClassicScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False),
	(linear_market.MultiCompetitorScenario, actorcritic_agent.DiscreteACALinear, False),
	(linear_market.MultiCompetitorScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(linear_market.MultiCompetitorScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomy, True),
	(circular_market.CircularEconomyMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.DiscreteACACircularEconomyRebuy, True),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, False),
	(circular_market.CircularEconomyRebuyPriceMonopolyScenario, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, True),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.DiscreteACACircularEconomyRebuy, False),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentFixedOneStd, True),
	(circular_market.CircularEconomyRebuyPriceOneCompetitor, actorcritic_agent.ContinuosActorCriticAgentEstimatingStd, False)
]


@pytest.mark.parametrize('marketplace, agent, verbose', test_scenarios)
def test_training_configurations(marketplace, agent, verbose):
	with patch('alpha_business.rl.training.SummaryWriter'), \
		patch('alpha_business.rl.actorcritic_agent.ActorCriticAgent.save'):
		ActorCriticTrainer(marketplace, agent, log_dir_prepend='test_').train_agent(
			verbose=verbose,
			number_of_training_steps=120,
			total_envs=64)
