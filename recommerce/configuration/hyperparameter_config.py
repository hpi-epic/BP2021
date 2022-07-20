import json
import os

from attrdict import AttrDict

from recommerce.configuration.json_configurable import JSONConfigurable
from recommerce.configuration.path_manager import PathManager
from recommerce.configuration.utils import get_class
from recommerce.market.sim_market import SimMarket
from recommerce.market.vendors import Agent


class HyperparameterConfigValidator():
	def __str__(self) -> str:
		"""
		This overwrites the internal function that get called when you call `print(class_instance)`.

		Instead of printing the class name, prints the instance variables as a dictionary.

		Returns:
			str: The instance variables as a dictionary.
		"""
		return f'{self.__class__.__name__}: {self.__dict__}'

	@classmethod
	def validate_config(cls, config: dict, checked_class: SimMarket or Agent) -> dict:
		"""
		Validate the given config dictionary.

		Args:
			config (dict): The config to validate and take the values from.
			checked_class (SimMarket or Agent): The relevant class for which the fields are to be checked.

		Returns:
			dict: A valid hyperparameter config dict.
		"""
		demanded_fields = [field for field, _, _ in checked_class.get_configurable_fields()]
		config = cls._validate_keys(config, demanded_fields)
		cls._check_types(config, checked_class.get_configurable_fields())
		cls._check_rules(config, checked_class.get_configurable_fields())
		return config

	@classmethod
	def _validate_keys(cls, config: dict, demanded_fields: list) -> dict:
		"""
		Checks if only valid keys were provided.

		Args:
			config (dict): The config which should contain all values in demanded_fields.
			demanded_fields (list): The list containing all values that should be contained in config.

		Returns:
			dict: The config dictionary containing only valid keys.
		"""
		config_keys = set(config.keys())
		# the config_type key is completely optional as it is only used for webserver validation, so we don't prevent people from adding it
		if 'config_type' in config_keys:
			config_keys.remove('config_type')
		demanded_keys = set(demanded_fields)
		if config_keys != demanded_keys:
			missing_keys = demanded_keys.difference(config_keys)
			redundant_keys = config_keys.difference(demanded_keys)
			if missing_keys:
				assert False, f'your config is missing {missing_keys}'
			if redundant_keys:
				for key in redundant_keys:
					config.pop(key)
		return config

	@classmethod
	def _check_types(cls, config: dict, configurable_fields: list, must_contain: bool = True) -> None:
		for field_name, type, _ in configurable_fields:
			try:
				assert isinstance(config[field_name], type), f'{field_name} must be a {type} but was {type(config[field_name])}'
			except KeyError as error:
				if must_contain:
					raise KeyError(f'Your config is missing the following required key: {field_name}') from error

	@classmethod
	def _check_rules(cls, config: dict, configurable_fields: list, must_contain: bool = True) -> None:
		for field_name, _, rule in configurable_fields:
			if rule is not None:
				assert callable(rule)
				check_method, error_string = rule(field_name)
				try:
					assert check_method(config[field_name]), error_string
				except KeyError as error:
					if must_contain:
						raise KeyError(f'Your config is missing the following required key: {field_name}') from error


class HyperparameterConfigLoader():

	@classmethod
	def load(cls, filename: str, checked_class: SimMarket or Agent) -> AttrDict:
		"""
		Load the market configuration json file from the `configuration_files` folder, validate all keys and return an AttrDict instance.
		This can only be done after the relevant `environment_config` has been loaded, if both are needed, as the checked_class needs to be known.

		Args:
			filename (str): The name of the json file containing the configuration values.
				Must be located in the `configuration_files` directory in the user's datapath folder.
			checked_class (SimMarket or Agent): The relevant class for which the fields are to be checked.

		Returns:
			AttrDict: An Arribute Dict containing the hyperparameters.
		"""
		# In case the class is still in string format, extract it
		if issubclass(checked_class, str):
			checked_class = get_class(checked_class)

		assert issubclass(checked_class, (SimMarket, Agent)), f'the provided checked_class must be a subclass of SimMarket \
			if the config is a market_config or of Agent if it is an rl_config: {checked_class}'
		assert issubclass(checked_class, JSONConfigurable), f'the provided checked_class must be a subclass of JSONConfigurable: {checked_class}'

		if not filename.endswith('.json'):
			filename += '.json'
		path = os.path.join(PathManager.user_path, 'configuration_files', filename)
		with open(path) as config_file:
			hyperparameter_config = json.load(config_file)

		HyperparameterConfigValidator.validate_config(config=hyperparameter_config, checked_class=checked_class)
		return AttrDict(hyperparameter_config)
