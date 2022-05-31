from django.db import models

from .abstract_config import AbstractConfig


class AgentsConfig(AbstractConfig, models.Model):
	def as_list(self) -> dict:
		referencing_agents = self.agentconfig_set.all()
		return [agent.as_dict() for agent in referencing_agents]
