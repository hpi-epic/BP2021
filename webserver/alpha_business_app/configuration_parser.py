from .models.config import Config, remove_none_values_from_dict


class ConfigurationParser():
	def __init__(self, _config: Config = None) -> None:
		self.config = _config

	def flat_dict_to_hierarchical_config_dict(self, flat_dict: dict) -> dict:
		# get all environment parameter
		environment_parameter = self._get_items_key_starts_with(flat_dict, 'environment-')
		# get all hyperparameters
		hyperparameter = self._get_items_key_starts_with(flat_dict, 'hyperparameter-')

		return {
			'environment': self._flat_environment_to_hierarchical(environment_parameter),
			'hyperparameter': self._flat_hyperparameter_to_hierarchical(hyperparameter)
			}

	def _flat_environment_to_hierarchical(self, flat_dict: dict) -> dict:
		# get all agents components form dict
		raw_agents_dict = self._get_items_key_starts_with(flat_dict, 'agents-')
		# remove the agents component from final dict
		environment_dict = self._substract_dicts(flat_dict, raw_agents_dict, 'agents-')
		# take first list element and remove empty values
		environment_dict = self._first_list_element_without_empty(environment_dict)
		# add parsed agents
		environment_dict['agents'] = self._flat_agents_to_hierarchical(raw_agents_dict)
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

	# HELPER
	def _get_items_key_starts_with(self, dict_with_complete_keyword: dict, keyword_part: str) -> dict:
		return {k[len(keyword_part):]: v for k, v in dict_with_complete_keyword.items() if k.startswith(keyword_part)}

	def _substract_dicts(self, dict1: dict, dict2: dict, keyword_extension: str = '') -> dict:
		dict2_with_extension = {keyword_extension + k: v for k, v in dict2.items()}
		return {k: v for k, v in dict1.items() if k not in dict2_with_extension}

	def _first_list_element_without_empty(self, dict1: dict) -> dict:
		return {k: v[0] for k, v in dict1.items() if v[0] != ''}
