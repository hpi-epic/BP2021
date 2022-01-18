import pytest

import rl.actorcritic_agent as a2c_agent

abstract_agent_classes_testcases = [
	a2c_agent.ActorCriticAgent,
	a2c_agent.DiscreteActorCriticAgent
]


@pytest.mark.parametrize('a2c_agent', abstract_agent_classes_testcases)
def test_abstract_agent_classes(a2c_agent):
	with pytest.raises(TypeError):
		a2c_agent()
