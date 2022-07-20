from django.db import models

from ..utils import remove_none_values_from_dict
from .abstract_config import AbstractConfig
from .rl_config import RlConfig
from .sim_market_config import SimMarketConfig


class HyperparameterConfig(AbstractConfig, models.Model):
	"""
	This class represents the `hyperparameter` part of our configuration file.
	It contains both `sim_market` config and `rl`config.
	"""
	rl = models.ForeignKey('alpha_business_app.RLConfig', on_delete=models.CASCADE, null=True)
	sim_market = models.ForeignKey('alpha_business_app.SimMarketConfig', on_delete=models.CASCADE, null=True)

	def as_dict(self) -> dict:
		sim_market_dict = self.sim_market.as_dict() if self.sim_market is not None else {'sim_market': None}
		rl_dict = self.rl.as_dict() if self.rl is not None else {'rl': None}
		return remove_none_values_from_dict({
			'rl': rl_dict,
			'sim_market': sim_market_dict
		})
