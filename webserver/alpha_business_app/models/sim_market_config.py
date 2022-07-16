from django.db import models

from .abstract_config import AbstractConfig


class SimMarketConfig(AbstractConfig, models.Model):
	"""
	This class encapsulates the database table for all sim_market parameters.
	It can be auto generated, using `on_recommerce_change`.
	This will set all fields to currently needed fields by the `recommerce` package
	"""
	common_state_visibility = models.BooleanField(null=True, default=None)
	episode_length = models.IntegerField(null=True, default=None)
	max_price = models.IntegerField(null=True, default=None)
	max_quality = models.IntegerField(null=True, default=None)
	max_storage = models.IntegerField(null=True, default=None)
	number_of_customers = models.IntegerField(null=True, default=None)
	opposite_own_state_visibility = models.BooleanField(null=True, default=None)
	production_price = models.IntegerField(null=True, default=None)
	reward_mixed_profit_and_difference = models.BooleanField(null=True, default=None)
	storage_cost_per_product = models.FloatField(null=True, default=None)
