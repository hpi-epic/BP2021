# This file can be used to write own config files.
# It should be executed before running migrations.
# When using this file or changing the implementation,
# please keep in mind, that this is a potential security risk

import os

from utils import get_structure_with_types_of, to_config_class_name


class ConfigModelWriter:
	"""
	This class can be used to write various files concerning changing sim market and rl models in recommerce.
	It should be used bevor server startup and migration on the change of the recommerce package.
	Please always double check the generated files when deploying in production.
	"""
	def __init__(self, top_level: str, second_level: str = None) -> None:
		self.whitespace = '\t'
		self.top_level = top_level
		self.second_level = second_level
		self.name = second_level if second_level else top_level
		self.class_name = to_config_class_name(self.name)

	def write_files(self) -> None:
		"""
		Writes Model file and template file.
		"""
		self.write_model_file()
		self.write_template()

	def write_model_file(self) -> None:
		"""
		Writes model file for current self.class_name.
		Gets all possible attributes from the recommerce package first and writes valid Django model code afterwards.
		"""
		print(f'{self._warning()}WARNING: This action will override the {self.class_name} model file.{self._end()}')
		print('Press enter to continue, Press any key to skip this file')
		if input():
			print(f'skipping {self.class_name} model file')
			return
		# imports
		lines = ['from django.db import models', '', 'from .abstract_config import AbstractConfig', '']
		# class definition
		lines += [f'class {self.class_name}(AbstractConfig, models.Model):']
		lines += [
			'\t"""',
			f'\tThis class encapsulates the database table for all {self.name} parameters.',
			'\tIt can be auto generated, using `on_recommerce_change`.',
			'\tThis will set all fields to currently needed fields by the `recommerce` package',
			'\t"""'
		]
		# fields
		attributes = get_structure_with_types_of(self.top_level, self.second_level)
		for attr in sorted(attributes, key=lambda tup: tup[0]):
			django_class = str(attr[1]).rsplit('.')[-1][:-2]
			additional_attributes = self._get_additional_attributes(django_class)
			lines += [f'{self.whitespace}{attr[0]} = models.{django_class}(null=True, default=None{additional_attributes})']
		path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f'{self.name}_config.py')
		self._write_lines_to_file(path_to_file, lines)

	def write_template(self) -> None:
		"""
		Writes template file for current self.class_name.
		Gets all possible attributes from the recommerce package first and writes valid html code afterwards.
		"""
		print(f'{self._warning()}WARNING: This action will override the {self.name} template file.{self._end()}')
		print('Press enter to continue, Press any key to skip this file')
		if input():
			print(f'skipping {self.class_name} template file')
			return
		lines = [
			'<div class="accordion-item">',
			'\t<h2 class="accordion-header">',
			'\t\t<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"',
			f'\t\t\tdata-bs-target="#collapse{self.class_name}" aria-expanded="false" aria-controls="collapse{self.class_name}">',
			f'\t\t\t{self._visual_name()}',
			'\t\t</button>',
			'\t</h2>',
			f'\t<div id="collapse{self.class_name}" class="accordion-collapse collapse">',
			'\t\t<div class="accordion-body">',
			'\t\t\t{% load static %}'
		]
		attributes = get_structure_with_types_of(self.top_level, self.second_level)
		for attr in sorted(attributes, key=lambda tup: tup[0]):
			input_type = self._to_html_input_type(attr[1])
			lines += self._row_lines_for(attr[0], input_type)

		lines += [
			'\t\t</div>',
			'\t</div>',
			'</div>',
			''
		]
		file_dir = os.path.dirname(os.path.abspath(__file__))
		path_to_file = os.path.join(file_dir, os.pardir, 'templates', 'configuration_items', f'{self.name}.html')
		self._write_lines_to_file(path_to_file, lines)

	def _row_lines_for(self, keyword: str, input_type: str) -> list:
		"""
		Retuns a list of lines with valid html code for one hyperparameter attribut aka a row in the template.

		Args:
			keyword (str): hyperparameter keyword
			input_type (str): html type of input field i.e. number or text

		Returns:
			list: of lines containing valid html code
		"""
		visual_keyword = keyword.replace('_', ' ')
		lines = [
			f'<div class="row p-2" id="hyperparameter-{self.name}-{keyword}">',
			'\t<div class="col-6">',
			'\t\t{% if error_dict.' + keyword + ' %}',
			'\t\t\t<img src="{% static \'icons/warning.svg\' %}" width="18px" title="{{error_dict.' + keyword + '}}"></img>',
			'\t\t{% endif %}',
			f'\t\t{visual_keyword}',
			'\t</div>',
			'\t<div class="col-6">',
			'\t\t<input type="' + input_type + '" class="form-control {% if error_dict.max_storage %} bc-error-field {% endif %}"',
			'\t\t\tmin="0" step="any" value="{{prefill.' + keyword + '}}" name="' + self.top_level + '-' + self.name + '-' + keyword + '">',
			'\t</div>',
			'</div>'
		]
		return [f'\t\t\t{line}' for line in lines]

	def _get_additional_attributes(self, django_class: str) -> str:
		"""
		At the moment only Charfield needs some extra parameters

		Args:
			django_class (str): a Django class as string

		Returns:
			str: possibl extra arguments, that need to be passed when initializing this field in Django code
		"""
		if 'CharField' in django_class:
			return ', max_length=100'
		return ''

	def _to_html_input_type(self, django_type: str) -> str:
		"""Convrets a Django datatype into an html datatype

		Args:
			django_type (str): a Django class as string

		Returns:
			str: either 'number' or 'text'
		"""
		if 'Int' or 'Float' in django_type:
			return 'number'
		else:
			return 'text'

	def _visual_name(self) -> str:
		"""
		The name that should be displayed to the user in template

		Raises:
			NotImplementedError: implement other names than rl or sim_market first

		Returns:
			str: need visual names
		"""
		if self.name == 'rl':
			return 'RL'
		elif self.name == 'sim_market':
			return 'Sim Market'
		else:
			raise NotImplementedError

	def _warning(self) -> str:
		return '\033[93m'

	def _write_lines_to_file(self, path_to_file: str, lines: list) -> None:
		"""
		Writes all collected lines into a file

		Args:
			path_to_file (str): path to the file that should be written to
			lines (list): lines that should be written
		"""
		print(f'Writing class definition of {self.class_name} to file.')
		with open(path_to_file, 'w') as config_file:
			config_file.write('\n'.join(lines))

	def _end(self) -> str:
		return '\033[0m'


ConfigModelWriter(top_level='hyperparameter', second_level='rl').write_files()
ConfigModelWriter(top_level='hyperparameter', second_level='sim_market').write_files()
