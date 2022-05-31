from django.db import models

from .abstract_config import AbstractConfig


class EnvironmentConfig(AbstractConfig, models.Model):
	agents = models.ForeignKey('alpha_business_app.AgentsConfig', on_delete=models.CASCADE, null=True)
	enable_live_draw = models.BooleanField(null=True)
	episodes = models.IntegerField(null=True)
	plot_interval = models.IntegerField(null=True)
	marketplace = models.CharField(max_length=150, null=True)
	task = models.CharField(max_length=14, choices=((1, 'training'), (2, 'agent_monitoring'), (3, 'exampleprinter')), null=True)
