from .models.config import *


class ConfigurationParser():
	def __init__(self, _config: Config = None) -> None:
		self.config = _config

	def flat_dict_to_hierarchical_config_dict(self, flat_dict: dict) -> dict:
		# prepare flat_dict, convert all numbers to int orr foat
		for key, value_list in flat_dict.items():
			converted_values = []
			for value in value_list:
				converted_values += [self._converted_to_int_or_float_if_possible(value)]
			flat_dict[key] = converted_values
		# get all environment parameter
		environment_parameter = self._get_items_key_starts_with(flat_dict, 'environment-')
		# get all hyperparameters
		hyperparameter = self._get_items_key_starts_with(flat_dict, 'hyperparameter-')

		return {
			'environment': self._flat_environment_to_hierarchical(environment_parameter),
			'hyperparameter': self._flat_hyperparameter_to_hierarchical(hyperparameter)
			}

	def parse_config(self, config_dict: dict):
		return self.parse_config_dict_to_datastructure('', config_dict)

	def parse_config_dict_to_datastructure(self, name, config_dict: dict):
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

	def _flat_environment_to_hierarchical(self, flat_dict: dict) -> dict:
		# get all agents components form dict
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
		final_dict = {}
		for agent_index in range(len(flat_dict['name'])):
			argument = flat_dict['argument'][agent_index] if flat_dict['argument'][agent_index] != '' else None
			agent_dict = {
				'agent_class': flat_dict['agent_class'][agent_index],
				'argument': argument
			}
			final_dict[flat_dict['name'][agent_index]] = remove_none_values_from_dict(agent_dict)
		return final_dict

	def _flat_hyperparameter_to_hierarchical(self, flat_dict: dict) -> dict:
		rl = self._get_items_key_starts_with(flat_dict, 'rl-')
		sim_market = self._get_items_key_starts_with(flat_dict, 'sim_market-')

		# take first list element and remove empty values
		rl = self._first_list_element_without_empty(rl)
		sim_market = self._first_list_element_without_empty(sim_market)

		return {'rl': rl, 'sim_market': sim_market}

	def _parse_agents_to_datastructure(self, agent_dict: dict) -> AgentsConfig:
		agents = AgentsConfig.objects.create()
		for agent_name, agent_parameters in agent_dict.items():
			agent_parameters['agents_config'] = agents
			agent_parameters['name'] = agent_name
			AgentConfig.objects.create(**agent_parameters)
		return agents

	def _create_object_from(self, class_name: str, parameters: dict):
		return globals()[class_name].objects.create(**parameters)

	# HELPER
	def _get_items_key_starts_with(self, dict_with_complete_keyword: dict, keyword_part: str) -> dict:
		return {k[len(keyword_part):]: v for k, v in dict_with_complete_keyword.items() if k.startswith(keyword_part)}

	def _substract_dicts(self, dict1: dict, dict2: dict, keyword_extension: str = '') -> dict:
		dict2_with_extension = {keyword_extension + k: v for k, v in dict2.items()}
		return {k: v for k, v in dict1.items() if k not in dict2_with_extension}

	def _first_list_element_without_empty(self, dict1: dict) -> dict:
		return {k: v[0] for k, v in dict1.items() if v[0] != ''}

	def _converted_to_int_or_float_if_possible(self, value: str):
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
