from django.db import models

from ..utils import remove_none_values_from_dict
from .abstract_config import AbstractConfig


class EnvironmentConfig(AbstractConfig, models.Model):
	"""
	This class represents the `environment` part of our configuration file.
	"""
	agents = models.ForeignKey('alpha_business_app.AgentsConfig', on_delete=models.CASCADE, null=True)
	separate_markets = models.BooleanField(null=True)
	episodes = models.IntegerField(null=True)
	plot_interval = models.IntegerField(null=True)
	marketplace = models.CharField(max_length=150, null=True)
	task = models.CharField(max_length=14, choices=((1, 'training'), (2, 'agent_monitoring'), (3, 'exampleprinter')), null=True)

	def as_dict(self) -> dict:
		agents_list = self.agents.as_list() if self.agents is not None else None
		return remove_none_values_from_dict({
			'separate_markets': self.separate_markets,
			'episodes': self.episodes,
			'plot_interval': self.plot_interval,
			'marketplace': self.marketplace,
			'task': self.task,
			'agents': agents_list
		})
