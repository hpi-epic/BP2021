from django.db import models

from .abstract_config import AbstractConfig


class AgentsConfig(AbstractConfig, models.Model):
	def as_list(self) -> dict:
		referencing_agents = self.agentconfig_set.all()
		return [agent.as_dict() for agent in referencing_agents]

	def as_dict(self) -> dict:
		assert False, 'This should not be implemented as agents are a list.'
