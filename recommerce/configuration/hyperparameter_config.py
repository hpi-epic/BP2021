import json
import os

from attrdict import AttrDict

# from recommerce.configuration.json_configurable import JSONConfigurable
from recommerce.configuration.path_manager import PathManager

# from recommerce.configuration.utils import get_class


class HyperparameterConfigValidator():
	@classmethod
	def get_required_fields(cls, dict_key) -> dict:
		"""
		THIS SHOULD BE REPLACED URGENTLY. USE THE INFORMATION PROVIDED BY CONFIGURABLE AGENTS TO GET THE REQUIRED FIELDS.
		"""
		if dict_key == 'top-dict':
			return {'rl': True, 'sim_market': True}
		elif dict_key == 'rl':
			return {
				'gamma': False,
				'batch_size': False,
				'replay_size': False,
				'learning_rate': False,
				'sync_target_frames': False,
				'replay_start_size': False,
				'epsilon_decay_last_frame': False,
				'epsilon_start': False,
				'epsilon_final': False
			}
		elif dict_key == 'sim_market':
			return {
				'max_storage': False,
				'episode_length': False,
				'max_price': False,
				'max_quality': False,
				'number_of_customers': False,
				'production_price': False,
				'storage_cost_per_product': False
			}
		else:
			raise AssertionError(f'The given level does not exist in a hyperparameter-config: {dict_key}')

	def __str__(self) -> str:
		"""
		This overwrites the internal function that get called when you call `print(class_instance)`.

		Instead of printing the class name, prints the instance variables as a dictionary.

		Returns:
			str: The instance variables as a dictionary.
		"""
		return f'{self.__class__.__name__}: {self.__dict__}'

	@classmethod
	def validate_config(cls, config: dict) -> None:
		"""
		Validate the given config dictionary.

		Args:
			config (dict): The config to validate and take the values from.
		"""
		demanded_fields = [field for field, _, _ in config['class'].get_configurable_fields()]
		cls._check_given_config_is_subset_of_demanded(config, demanded_fields)
		cls._check_demanded_config_is_subset_of_given(config, demanded_fields)
		cls._check_types(config, config['class'].get_configurable_fields())
		cls._check_rules(config, config['class'].get_configurable_fields())

	@classmethod
	def _check_given_config_is_subset_of_demanded(cls, config: dict, demanded_fields: list) -> None:
		for key in config:
			assert key == 'class' or key in demanded_fields, f'your config provides {key} which was not demanded'

	@classmethod
	def _check_demanded_config_is_subset_of_given(cls, config: dict, demanded_fields: list) -> None:
		for key in demanded_fields:
			assert key in config, f'your config is missing {key}'

	@classmethod
	def _check_types(cls, config: dict, configurable_fields: list) -> None:
		for field_name, type, _ in configurable_fields:
			assert isinstance(config[field_name], type), f'{field_name} must be a {type} but was {type(config[field_name])}'

	@classmethod
	def _check_rules(cls, config: dict, configurable_fields: list) -> None:
		for field_name, _, rule in configurable_fields:
			if rule is not None:
				if not isinstance(rule, tuple):
					assert callable(rule)
					check_method, error_string = rule(field_name)
				else:
					check_method, error_string = rule
				assert check_method(config[field_name]), error_string


class HyperparameterConfigLoader():
	@classmethod
	def load(cls, filename: str) -> AttrDict:
		"""
		Load the configuration json file from the `configuration_files` folder, validate all keys and retruning an AttrDict instance
		without top level keys.

		Args:
			filename (str): The name of the json file containing the configuration values.
				Must be located in the `configuration_files` directory in the user's datapath folder.

		Returns:
			AttrDict: An Arribute Dict containing the hyperparameters.
		"""
		filename += '.json'
		path = os.path.join(PathManager.user_path, 'configuration_files', filename)
		with open(path) as config_file:
			config = json.load(config_file)
		# assert 'class' in config, f"Every config json must contain a 'class' key, but {filename} does not."
		# config['class'] = get_class(config['class'])
		# assert issubclass(config['class'], JSONConfigurable), f"The class {config['class']} must be a subclass of JSONConfigurable."
		# print(config)
		# HyperparameterConfigValidator.validate_config(config)
		# config.pop('class')
		return AttrDict(config)
