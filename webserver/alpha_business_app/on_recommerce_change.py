# This file can be used to write own config files.
# It should be executed before running migrations.
# When using this file or changing the implementation,
# please keep in mind, that this is a potential security rist

import os

from utils import get_structure_with_types_of, to_config_class_name


class ConfigModelWriter:
	def __init__(self, top_level: str, second_level: str = None) -> None:
		self.whitespace = '\t'
		self.top_level = top_level
		self.second_level = second_level
		# self.structure_dict = get_structure_of(top_level, second_level)
		self.name = second_level if second_level else top_level

	def write_file(self) -> None:
		# imports
		lines = ['from django.db import models', '', 'from .abstract_config import AbstractConfig', '']
		# class definition
		lines += [f'class {to_config_class_name(self.name)}(AbstractConfig, models.Model):']
		# fields
		attributes = get_structure_with_types_of(self.top_level, self.second_level)
		for attr in attributes:
			django_class = str(attr[1]).rsplit('.')[-1][:-1]
			lines += [f'{self.whitespace}{attr[0]} = models.{django_class}(null=True, default=None)']
		path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f'{self.name}_config.py')
		# write to file
		with open(path_to_file, 'w') as config_file:
			config_file.write('\n'.join(lines))


ConfigModelWriter(top_level='hyperparameter', second_level='rl').write_file()
