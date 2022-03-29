from .models.config import Config


class ConfigMerger():
	def __init__(self) -> None:
		self.error_dict = Config.get_empty_structure_dict()

	def merge_config_objects(self, config_object_ids: list) -> tuple:
		"""
		merge a list of config objects given by their id.

		Args:
			config_object_ids (list): The id's of the config objects that should be merged.

		Returns:
			tuple (dict, dict): the final merged dict and the error dict whith the latest error
		"""
		configuration_objects = [Config.objects.get(id=config_id) for config_id in config_object_ids]
		configuration_dicts = [config.as_dict() for config in configuration_objects]
		# get initial empty dict to merge into
		final_config = Config.get_empty_structure_dict()
		for config in configuration_dicts:
			final_config = self._merge_config_into_base_config(final_config, config)
		return final_config, self.error_dict

	def _merge_config_into_base_config(self, base_config: dict, merging_config: dict, current_config_path: str = '') -> dict:
		"""
		merges one config dict recursively into a base_config dict.

		Args:
			base_config (dict): the config that will be merged into
			merging_config (dict): the config that should be merged
			current_config_path (str, optional): keep track of the current config path for the error dict. Defaults to ''.

		Returns:
			dict: a final merged config
		"""
		# get contained dicts for recursion
		contained_dicts_merge = [(key, value) for key, value in merging_config.items() if type(value) == dict]
		# get contained values for updating
		contained_values_merge = [(key, value) for key, value in merging_config.items() if type(value) != dict and value is not None]

		for key, sub_dict in contained_dicts_merge:
			if key == 'agents':
				base_config[key] = self._merge_agents_into_base_agents(base_config[key], sub_dict)
				continue
			new_config_path = f'{current_config_path}-{key}' if current_config_path else key
			base_config[key] = self._merge_config_into_base_config(base_config[key], sub_dict,  new_config_path)

		# update values
		for key, value in contained_values_merge:
			if base_config[key] is not None and base_config[key] != value:
				# configs differ, we need to insert this in our error dict
				error_message = f'changed {current_config_path} {key} from {base_config[key]} to {value}'
				self._update_error_dict(current_config_path.split('-') + [key], error_message)
			base_config[key] = value

		return base_config

	def _merge_agents_into_base_agents(self, base_agent_config: dict, merge_agent_config: dict) -> dict:
		"""
		Merges an agents config part into a base agents config part. It will be checked if two of the merged agents have the same name.

		Args:
			base_agent_config (dict): the config that will be merged into
			merge_agent_config (dict): the config that should be merged

		Returns:
			dict: a final merged agents config
		"""
		for agent_name, _ in merge_agent_config.items():
			if agent_name in base_agent_config:
				self._update_error_dict(['environment', 'agents'], f'multiple {agent_name}')
		return {**base_agent_config, **merge_agent_config}

	def _update_error_dict(self, key_words: list, update_message: str) -> None:
		"""
		helper function, that updates a value in the error dict given by the list of key words

		Args:
			key_words (list): 'path' to the keyword that should be updated
			update_message (str): message that should be written to the keyword
		"""
		if len(key_words) == 2:
			self.error_dict[key_words[0]][key_words[1]] = update_message
		elif len(key_words) == 3:
			self.error_dict[key_words[0]][key_words[1]][key_words[2]] = update_message
		else:
			# our config is (should) always be at least two keywords deep
			assert False
