from recommerce.configuration.utils import get_class

from .utils import convert_python_type_to_input_type


def get_agent_hyperparameter(agent: str, formdata: dict) -> list:

	# get all fields that are possible for this agent
	agent_class = get_class(agent)
	agent_specs = agent_class.get_configurable_fields()

	# we want to keep values already inside the html, so we need to parse existing html
	parameter_values = _convert_form_to_value_dict(formdata)
	# convert parameter into special list format for view
	all_parameter = []
	for spec in agent_specs:
		this_parameter = {}
		this_parameter['name'] = spec[0]
		this_parameter['input_type'] = convert_python_type_to_input_type(spec[1])
		this_parameter['prefill'] = _get_value_from_dict(spec[0], parameter_values)
		all_parameter += [this_parameter]
	return all_parameter


def get_rl_parameter_prefill(prefill: dict, error: dict) -> list:
	# returns list of dictionaries
	all_parameter = []
	for key, value in prefill.items():
		this_parameter = {}
		this_parameter['name'] = key
		this_parameter['prefill'] = value if value else ''
		this_parameter['error'] = error[key] if error[key] else ''
		all_parameter += [this_parameter]
	return all_parameter


def _convert_form_to_value_dict(config_form) -> dict:
	final_values = {}
	for index in range((len(config_form) - 2) // 2):
		current_name = config_form[f'formdata[{index}][name]']
		current_value = config_form[f'formdata[{index}][value]']
		if 'hyperparameter-rl' in current_name:
			final_values[current_name.replace('hyperparameter-rl-', '')] = current_value
	return final_values


def _get_value_from_dict(key, value_dict) -> dict:
	try:
		return value_dict[key]
	except KeyError:
		return ''
