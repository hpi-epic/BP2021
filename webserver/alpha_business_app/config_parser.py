from .models.config import *


class ConfigFlatDictParser():
	"""
	This class can parse a flat config dict to a hierarchical config dict.
	"""

	def flat_dict_to_hierarchical_config_dict(self, flat_dict: dict) -> dict:
		"""
		Parses a flat config dictionary to hierarchical dictionary.
		The flat dictionary should have the hierarchical path in the keys, separated by '-'

		Args:
			flat_dict (dict): a dictionary containing key value pairs from a config.
				Keys must have a keypath for hierarchical config separated by '-'

		Returns:
			dict: hierarchical config dict
		"""
		# prepare flat_dict, convert all numbers to int or float
		for key, value_list in flat_dict.items():
			converted_values = [self._converted_to_int_or_float_if_possible(value) for value in value_list]
			# This happens if the user provides the same agent name twice, since they are keys in the environment_config.json
			if flat_dict[key]:
				assert False, 'At least two agents have the same name/key'
			flat_dict[key] = converted_values

		# get all environment parameters
		environment_parameter = self._get_items_key_starts_with(flat_dict, 'environment-')
		# get all hyperparameters
		hyperparameter = self._get_items_key_starts_with(flat_dict, 'hyperparameter-')

		return {
			'environment': self._flat_environment_to_hierarchical(environment_parameter),
			'hyperparameter': self._flat_hyperparameter_to_hierarchical(hyperparameter)
			}

	def _flat_environment_to_hierarchical(self, flat_dict: dict) -> dict:
		"""
		Parses the environment part of the flat dict to a hierarchical environment dict.

		Args:
			flat_dict (dict): a flat dict with keywords that only belong to the environment config.

		Returns:
			dict: hierarchical dict of the environment config.
		"""
		# get all agents components from dict
		raw_agents_dict = self._get_items_key_starts_with(flat_dict, 'agents-')
		# remove the agents component from final dict
		environment_dict = self._substract_dicts(flat_dict, raw_agents_dict, 'agents-')
		# take first list element and remove empty values
		environment_dict = self._first_list_element_without_empty(environment_dict)
		# add parsed agents
		environment_dict['agents'] = self._flat_agents_to_hierarchical(raw_agents_dict)
		# add enable_live_draw if exists#
		environment_dict['enable_live_draw'] = 'enable_live_draw' in flat_dict
		return environment_dict

	def _flat_agents_to_hierarchical(self, flat_dict: dict) -> dict:
		"""
		Parses the agents from a flat dict to a hierarchical dictionary.

		Args:
			flat_dict (dict): dicttionary containing only keys that belong to agents.

		Returns:
			dict: a hierarchical agents dictionary.
		"""
		final_dict = {}
		for agent_index in range(len(flat_dict['name'])):
			argument = flat_dict['argument'][agent_index]
			agent_dict = {
				'agent_class': flat_dict['agent_class'][agent_index],
				'argument': argument
			}
			final_dict[flat_dict['name'][agent_index]] = remove_none_values_from_dict(agent_dict)
		return final_dict

	def _flat_hyperparameter_to_hierarchical(self, flat_dict: dict) -> dict:
		"""
		Parses the hyperparameter part of the flat dict to a hierarchical hyperparameter dict.

		Args:
			flat_dict (dict): a flat dict with keywords that only belong to the hyperparameter config.

		Returns:
			dict: hierarchical dict of the hyperparameter config.
		"""
		rl = self._get_items_key_starts_with(flat_dict, 'rl-')
		sim_market = self._get_items_key_starts_with(flat_dict, 'sim_market-')

		# take first list element and remove empty values
		rl = self._first_list_element_without_empty(rl)
		sim_market = self._first_list_element_without_empty(sim_market)

		return {'rl': rl, 'sim_market': sim_market}

	# HELPER
	def _get_items_key_starts_with(self, dict_with_complete_keyword: dict, keyword_part: str) -> dict:
		"""
		Returns all key value pairs as dict where the key starts with specific keyword.
		The key will be without the specified keyword in the resulting dictionary.

		Args:
			dict_with_complete_keyword (dict): dictionary which should be filtered.
			keyword_part (str): The identifier of the resulting dictionary.

		Returns:
			dict: All key value pairs that started with the specified keyword.
		"""
		return {key[len(keyword_part):]: value for key, value in dict_with_complete_keyword.items() if key.startswith(keyword_part)}

	def _substract_dicts(self, dict1: dict, dict2: dict, keyword_extension: str = '') -> dict:
		"""
		substracts `dict2` from `dict1`. `Dict2`can be extended with an optional keyword for that.

		Args:
			dict1 (dict): the dictionary that should be substracted from (minuend)
			dict2 (dict): the dictionary that should be subtracted (subtrahend)
			keyword_extension (str, optional): A keyword that will be placed in front of each key in `dict2` before substracting. Defaults to ''.

		Returns:
			dict: containing all keys and values from `dict1` that are not in `dict2`
		"""
		dict2_with_extension = {keyword_extension + key: value for key, value in dict2.items()}
		return {key: value for key, value in dict1.items() if key not in dict2_with_extension}

	def _first_list_element_without_empty(self, dict1: dict) -> dict:
		"""
		For a dictionary with list values, this will return you a dictionary with the same keys and as value the first element of the list.

		Args:
			dict1 (dict): a dictionary with only list values.

		Returns:
			dict: dictionary with the keys and the first list element as values
		"""
		return {key: value[0] for key, value in dict1.items() if value[0] != ''}

	def _converted_to_int_or_float_if_possible(self, value: str):
		"""
		if possible the input string will be converted to float or int. if not the string will be returned.

		Args:
			value (str): the value as string that should be converted.

		Returns:
			int, float or string: the value converted if possible
		"""
		try:
			float(value)
		except ValueError:
			# it is a string, we can't change it
			return value
		try:
			int(value)
		except ValueError:
			# it is a float
			return float(value)
		return int(value)


class ConfigModelParser():
	def parse_config(self, config_dict: dict) -> Config:
		"""
		Parses an hierarchical dict to the datastructure.

		Args:
			config_dict (dict): a hierarchical config dict

		Returns:
			Config: a config object containing the parsed config dict as objects
		"""
		return self.parse_config_dict_to_datastructure('', config_dict)

	def parse_config_dict_to_datastructure(self, name: str, config_dict: dict):
		"""
		Parses an hierarchical dictionary recursively to the datastructure.

		Args:
			name (str): the key of the current dict.
			config_dict (dict): part of the config dict belonging to the keyword name.

		Returns:
			an model instance of the config dict.
		"""
		if config_dict == {}:
			return

		if name == 'agents':
			# since django does only support many-to-one relationships (not one-to-many),
			# we need to parse the agents slightly different, to be able to reference many agents with the agents keyword
			return self._parse_agents_to_datastructure(config_dict)
		# get all key value pairs, that contain another dict
		containing_dict = [(name, value) for name, value in config_dict.items() if type(value) == dict]
		# loop through of these pairs, in order to parse these dictionaries and add
		# the parsed sub-element to the current element
		sub_elements = []
		for keyword, config in containing_dict:
			sub_elements += [(keyword, self.parse_config_dict_to_datastructure(keyword, config))]

		# get all elements that do not contain another dictionary
		not_containing_dict = dict([(name, value) for name, value in config_dict.items() if type(value) != dict])

		# add the sub-elements to the dictionary with the other key value pairs not containing another dictionary
		for keyword, model_instance in sub_elements:
			not_containing_dict[keyword] = model_instance

		# figure out which config object to create and return the created objects
		config_class = to_config_class_name(name)
		return self._create_object_from(config_class, not_containing_dict)

	def _parse_agents_to_datastructure(self, agent_dict: dict) -> AgentsConfig:
		"""
		Parses the part of the config for the keywort `agents`.

		Args:
			agent_dict (dict): the dictionary of agents.

		Returns:
			AgentsConfig: an instance of AgentsConfig multiple agents are referring to.
		"""
		agents = AgentsConfig.objects.create()
		for agent_name, agent_parameters in agent_dict.items():
			agent_parameters['agents_config'] = agents
			agent_parameters['name'] = agent_name
			AgentConfig.objects.create(**agent_parameters)
		return agents

	def _create_object_from(self, class_name: str, parameters: dict):
		"""
		Creates an instance of an imported class with the given parameters

		Args:
			class_name (str): class name of an imported class as string.
			parameters (dict): parameters for the object that should be created.

		Returns:
			an instance of the `class_name` with the parameters given.
		"""
		return globals()[class_name].objects.create(**parameters)
