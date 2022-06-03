from recommerce.configuration.utils import get_class

from .utils import convert_python_type_to_input_type


def get_agent_hyperparameter(agent: str, formdata: dict) -> list:
	"""
	Gets all hyperparameters for a specific agent in our list of dict format needed for the view.

	Args:
		agent (str): classname as string of a recommerce agent
		formdata (dict): content of the current configuration form

	Returns:
		list: of dict, the dicts contain the following values (currently needed by view):
			name: 		name of the hyperparameter
			input_type: html type for the input field e.g. number
			prefill: 	value that is already stored for this hyperparameter
	"""
	# get all fields that are possible for this agent
	agent_class = get_class(agent)
	agent_specs = agent_class.get_configurable_fields()

	# we want to keep values already inside the configuration form, so we need to parse existing html
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
	"""
	Converts a prefill and error dict to our list of dictionary format needed by view.

	Args:
		prefill (dict): 'rl' prefill dictionary
		error (dict): 'rl' error dictionary produced by merging config objects

	Returns:
		list: of dict, the dicts contain the following values (currently needed by view):
			name: 		name of the hyperparameter
			input_type: html type for the input field e.g. number
			prefill: 	value that is already stored for this hyperparameter
			error: 		error value for this parameter
	"""
	# returns list of dictionaries
	all_parameter = []
	for key, value in prefill.items():
		this_parameter = {}
		this_parameter['name'] = key
		this_parameter['prefill'] = value if value else ''
		this_parameter['error'] = error[key] if error[key] else ''
		all_parameter += [this_parameter]
	return all_parameter


def _convert_form_to_value_dict(config_form: dict) -> dict:
	"""
	Extracts the 'rl' part from the formdata dict as hierarchical dict.

	Args:
		config_form (dict): flat config form from the website

	Returns:
		dict: hierarchical rl dict with extracted values
	"""
	final_values = {}
	# the formdata is a flat dict, containing two values per config parameter, name and value
	# num_experiments and experiment_name are included in the form as well, but we do not consider those
	for index in range((len(config_form) - 2) // 2):
		current_name = config_form[f'formdata[{index}][name]']
		current_value = config_form[f'formdata[{index}][value]']
		if 'hyperparameter-rl' in current_name:
			final_values[current_name.replace('hyperparameter-rl-', '')] = current_value
	return final_values


def _get_value_from_dict(key: str, value_dict: dict) -> str:
	"""
	Save way to get either the key value of a key or ''.

	Args:
		key (str): key the value should be retrieved from
		value_dict (dict): dict the value should be retrieved from

	Returns:
		str: value of the key in dict or ''
	"""
	try:
		return value_dict[key]
	except KeyError:
		return ''
