from django.db import models

from .abstract_config import AbstractConfig
from .rl_config import RlConfig
from .sim_market_config import SimMarketConfig


class HyperparameterConfig(AbstractConfig, models.Model):
	rl = models.ForeignKey('alpha_business_app.RLConfig', on_delete=models.CASCADE, null=True)
	sim_market = models.ForeignKey('alpha_business_app.SimMarketConfig', on_delete=models.CASCADE, null=True)
