from .models.config import Config


class ConfigMerger():
	def __init__(self) -> None:
		self.epsilon = 0.001
		self.error_dict = Config.get_empty_structure_dict()

	def merge_config_objects(self, config_object_ids: list):
		configuration_objects = [Config.objects.get(id=config_id) for config_id in config_object_ids]
		configuration_dicts = [config.as_dict() for config in configuration_objects]
		final_config = Config.get_empty_structure_dict()
		for config in configuration_dicts:
			final_config = self._merge_config_into_base_config(final_config, config)
		return final_config, self.error_dict

	def _merge_config_into_base_config(self, base_config: dict, merging_config: dict, current_config_path: str = '') -> dict:
		# get contained dicts for recursion
		contained_dicts_merge = [(k, v) for k, v in merging_config.items() if type(v) == dict]
		# get contained values for updating
		contained_values_merge = [(k, v) for k, v in merging_config.items() if type(v) != dict and v is not None]

		for key, sub_dict in contained_dicts_merge:
			if key == 'agents':
				base_config[key] = self.merge_agents(base_config[key], sub_dict)
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

	def merge_agents(self, base_agent_config: dict, merge_agent_config: dict) -> dict:
		for agent_name, _ in merge_agent_config.items():
			if agent_name in base_agent_config:
				self._update_error_dict(['environment', 'agents'], f'multiple {agent_name}')
		return {**base_agent_config, **merge_agent_config}

	def _update_error_dict(self, key_words: list, update_message: str) -> None:
		if len(key_words) == 1:
			# our config is always at least two keywords deep
			assert False
		elif len(key_words) == 2:
			self.error_dict[key_words[0]][key_words[1]] = update_message
		elif len(key_words) == 3:
			self.error_dict[key_words[0]][key_words[1]][key_words[2]] = update_message
		else:
			assert False
