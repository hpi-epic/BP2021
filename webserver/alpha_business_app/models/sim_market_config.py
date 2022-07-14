from django.db import models

from .abstract_config import AbstractConfig


class SimMarketConfig(AbstractConfig, models.Model):
	production_price = models.IntegerField(null=True, default=None)
	storage_cost_per_product = models.FloatField(null=True, default=None)
	max_storage = models.IntegerField(null=True, default=None)
	max_quality = models.IntegerField(null=True, default=None)
	number_of_customers = models.IntegerField(null=True, default=None)
	episode_length = models.IntegerField(null=True, default=None)
	reward_mixed_profit_and_difference = models.CharField(null=True, default=None, max_length=100)
	max_price = models.IntegerField(null=True, default=None)
	opposite_own_state_visibility = models.CharField(null=True, default=None, max_length=100)
	common_state_visibility = models.CharField(null=True, default=None, max_length=100)
