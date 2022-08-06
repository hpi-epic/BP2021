from django.db import models

from ..utils import get_structure_dict_for, remove_none_values_from_dict, to_config_keyword


class AbstractConfig():
	"""
	This class is an abstract configuration, implementing important functions for parsing configurations.
	All configuration classes need to be child classes of this class.
	"""
	def as_dict(self) -> dict:
		config_field_values = vars(self)
		resulting_dict = {}
		for key, value in config_field_values.items():
			if key.startswith('_') or 'id' in key:
				continue
			resulting_dict[key] = value
		return remove_none_values_from_dict(resulting_dict)

	@classmethod
	def get_empty_structure_dict(cls: models.Model) -> dict:
		return get_structure_dict_for(to_config_keyword(str(cls)))
