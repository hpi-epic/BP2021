from django.db import models

from .abstract_config import AbstractConfig


class AgentsConfig(AbstractConfig, models.Model):
	"""
	This class encapsulates the database table for the `agents` component of our configuration file.
	It collects all `AgentConfig`, because they can refere to an instance of the class.
	A `Config` object should only have one `AgentsConfig`.
	"""
	def as_list(self) -> dict:
		referencing_agents = self.agentconfig_set.all()
		return [agent.as_dict() for agent in referencing_agents]

	def as_dict(self) -> dict:
		assert False, 'This should not be implemented as agents are a list.'
