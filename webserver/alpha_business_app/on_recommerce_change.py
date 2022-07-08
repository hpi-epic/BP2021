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
		self.name = second_level if second_level else top_level
		self.class_name = to_config_class_name(self.name)

	def write_file(self) -> None:
		print(f'{self._warning()}WARNING: This action will override the {self.class_name} file.{self._end()}')
		print('Press enter to continue')
		input()
		# imports
		lines = ['from django.db import models', '', 'from .abstract_config import AbstractConfig', '']
		# class definition
		lines += [f'class {self.class_name}(AbstractConfig, models.Model):']
		# fields
		attributes = get_structure_with_types_of(self.top_level, self.second_level)
		for attr in attributes:
			django_class = str(attr[1]).rsplit('.')[-1][:-2]
			additional_attributes = self._get_additional_attributes(django_class)
			lines += [f'{self.whitespace}{attr[0]} = models.{django_class}(null=True, default=None{additional_attributes})']
		path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f'{self.name}_config.py')
		# write to file
		print(f'Writing class definition of {self.class_name} to file.')
		with open(path_to_file, 'w') as config_file:
			config_file.write('\n'.join(lines))

	def _get_additional_attributes(self, django_class: str) -> str:
		if 'CharField' in django_class:
			return ', max_length=100'
		return ''

	def _warning(self) -> str:
		return '\033[93m'

	def _end(self) -> str:
		return '\033[0m'


ConfigModelWriter(top_level='hyperparameter', second_level='rl').write_file()
ConfigModelWriter(top_level='hyperparameter', second_level='sim_market').write_file()
