from django.db import models

from .abstract_config import AbstractConfig


class SimMarketConfig(AbstractConfig, models.Model):
	max_storage = models.IntegerField(null=True)
	episode_length = models.IntegerField(null=True)
	max_price = models.IntegerField(null=True)
	max_quality = models.IntegerField(null=True)
	number_of_customers = models.IntegerField(null=True)
	production_price = models.IntegerField(null=True)
	storage_cost_per_product = models.FloatField(null=True)
