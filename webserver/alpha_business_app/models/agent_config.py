from django.db import models

from .abstract_config import AbstractConfig


class AgentConfig(AbstractConfig, models.Model):
	agents_config = models.ForeignKey('alpha_business_app.AgentsConfig', on_delete=models.CASCADE, null=True)
	name = models.CharField(max_length=100, default='')
	agent_class = models.CharField(max_length=100, null=True)
	argument = models.CharField(max_length=200, default='')
