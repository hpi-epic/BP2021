from django.contrib.auth.models import User
from django.db import models

from ..utils import remove_none_values_from_dict
from .abstract_config import AbstractConfig
from .environment_config import EnvironmentConfig
from .hyperparameter_config import HyperparameterConfig


class Config(AbstractConfig, models.Model):
	environment = models.ForeignKey('alpha_business_app.EnvironmentConfig', on_delete=models.CASCADE, null=True)
	hyperparameter = models.ForeignKey('alpha_business_app.HyperparameterConfig', on_delete=models.CASCADE, null=True)
	name = models.CharField(max_length=100, editable=False, default='')
	user = models.ForeignKey(User, on_delete=models.CASCADE, null=True,)

	def as_dict(self) -> dict:
		environment_dict = self.environment.as_dict() if self.environment is not None else None
		hyperparameter_dict = self.hyperparameter.as_dict() if self.hyperparameter is not None else None
		return remove_none_values_from_dict({'environment': environment_dict, 'hyperparameter': hyperparameter_dict})

	def is_referenced(self):
		# Query set is empty so we are not referenced by any container
		return bool(self.container_set.all())
