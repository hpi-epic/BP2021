# this is a temp file

from .models.config import *
from .models.config import get_config_field_names, to_config_class_name


def check_dict_keys(name: str, content: dict) -> tuple:
	if name == 'agents':
		status, error_msg = check_agents(content)
		if not status:
			return False, error_msg
		else:
			return True, ''
	containing_dict = [(name, value) for name, value in content.items() if type(value) == dict]

	for keyword, config in containing_dict:
		status, error_msg = check_dict_keys(keyword, config)
		if not status:
			return False, error_msg

	class_name = to_config_class_name(name)
	allowed_keys = get_config_field_names(globals()[class_name])
	used_keys = list(content.keys())
	difference = list(set(used_keys) - set(allowed_keys))
	if difference != []:
		return False, f'The keyword(s) {difference} are not allowed in {name}'

	return True, ''


def check_agents(content: dict) -> tuple:
	for agent_name, agent_parameters in content.items():
		allowed_keys = ['name', 'agents_config', 'argument', 'agent_class']
		used_keys = list(agent_parameters.keys())
		difference = list(set(used_keys) - set(allowed_keys))
		if difference != []:
			return False, f'The keyword(s) {difference} are not allowed in {agent_name}'
	return True, ''
