import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.configuration.environment_config import EnvironmentConfig
from recommerce.configuration.utils import get_class


def convert_python_type_to_input_type(to_convert) -> str:
	return 'number' if to_convert == float or to_convert == int else 'text'


def convert_python_type_to_django_type(to_convert: type) -> str:
	"""
	Converts standard python types, into a string of a Django model classes.
	At the moment float, and int are supported, the rest will be Charfield.

	Args:
		to_convert (type): standard python type ro be converted

	Returns:
		str: string of a corresponding Django model class
	"""
	from django.db import models
	if to_convert == float or (type(to_convert) == tuple and float in to_convert):
		return str(models.FloatField)
	elif to_convert == int:
		return str(models.IntegerField)
	else:
		return str(models.CharField)


def get_recommerce_marketplaces() -> list:
	"""
	Matches marketplaces of recommerce.market.circular.circular_sim_market and recommerce.market.linear.linear_sim_market,
	which contain one of the Keywords: Oligopoly, Duopoly, Monopoly

	Returns:
		list: tuple list for selection
	"""
	keywords = ['Monopoly', 'Duopoly', 'Oligopoly']
	# get all circular marketplaces
	circular_marketplaces = list(set(filter(lambda class_name: any(keyword in class_name for keyword in keywords), dir(circular_market))))
	circular_market_str = [f'recommerce.market.circular.circular_sim_market.{market}' for market in sorted(circular_marketplaces)]
	# get all linear marketplaces
	visible_linear_names = list(set(filter(lambda class_name: any(keyword in class_name for keyword in keywords), dir(linear_market))))
	linear_market_str = [f'recommerce.market.linear.linear_sim_market.{market}' for market in sorted(visible_linear_names)]

	return circular_market_str + linear_market_str


def get_recommerce_agents_for_marketplace(marketplace) -> list:
	return marketplace.get_possible_rl_agents()


def get_all_possible_rl_hyperparameter() -> set:
	"""
	Gets all hyperparameters for all possible recommerce agents

	Returns:
		set: of tuples, containing the hyperparameter name and the hyperparameter type
	"""
	all_marketplaces = get_recommerce_marketplaces()
	all_agents = []
	for marketplace_str in all_marketplaces:
		marketplace = get_class(marketplace_str)
		all_agents += get_recommerce_agents_for_marketplace(marketplace)

	return get_attributes(all_agents)


def get_not_possible_agent_hyperparameter(agent: str) -> tuple:
	"""
	Returns two lists, the first one being all rl hyperparameter that are possible for that agent,
	the second one being all possible rl hyperparamter.
	Both lists contain the parameter with the prefix 'hyperparameter-rl-' as string.

	Args:
		agent (str): the Agent we want to get the fields that are not possible of

	Returns:
		tuple: of lists, containing parameters the agent does not have and all possible rl parameter.
	"""
	prefix = 'hyperparameter-rl-'
	all_parameter = set([prefix + parameter[0] for parameter in get_all_possible_rl_hyperparameter()])
	specific_parameters = set([prefix + parameter[0] for parameter in get_class(agent).get_configurable_fields()])
	return list(all_parameter - specific_parameters), list(all_parameter)


def get_all_possible_sim_market_hyperparameter() -> set:
	"""
	Gets all hyperparameters for all possible recommerce markets

	Returns:
		set: of tuples, containing the hyperparameter name and the hyperparameter type
	"""
	all_marketplaces = get_recommerce_marketplaces()
	return get_attributes(all_marketplaces)


def get_attributes(all_classes: list) -> set:
	"""
	Calls `get_configurable_fields` and collects the name and the type of the returned fields in a list.

	Args:
		all_classes (list): list of strings of classes that implement `get_configurable_fields`

	Returns:
		set: of tuples, containing the attribute name and the attribute type
	"""
	all_attributes = []
	for class_str in all_classes:
		current_class = get_class(class_str)
		try:
			# we do not necessarily need to include the rule, as it is currently not used in the webserver
			all_attributes += [attribute[:2] for attribute in current_class.get_configurable_fields()]
		except NotImplementedError:
			print(f'please check the installation of the recommerce package!{current_class} does not implement "get_configurable_fields"')
	return set(all_attributes)


def get_structure_dict_for(keyword: str) -> dict:
	"""
	Will return a Dictionary of the complete structure (all possible fields) for one suitable keyword.

	Args:
		keyword (str): must be 'environment', 'hyperparameter', 'sim_market', 'rl', 'agents' or ''.
			'' means it will return the whole strucutre dict

	Returns:
		dict: general structure of the given keywords, values will always be None
	"""
	assert keyword in ['environment', 'hyperparameter', 'sim_market', 'rl', 'agents', ''], f'Your keyword {keyword} is not recognized.'
	environment_dict = EnvironmentConfig.get_required_fields('top-dict')
	environment_dict_with_none = {key: None for key in environment_dict.keys()}
	environment_dict_with_none['agents'] = []

	hyperparameter_dict_sim_market = {parameter[0]: None for parameter in get_all_possible_sim_market_hyperparameter()}
	hyperparameter_dict_rl = {parameter[0]: None for parameter in get_all_possible_rl_hyperparameter()}

	hyperparameter_dict = {
		'sim_market': hyperparameter_dict_sim_market,
		'rl': hyperparameter_dict_rl
	}

	structure_config_dict = {
		'environment': environment_dict_with_none,
		'hyperparameter': hyperparameter_dict
	}
	if keyword == 'environment':
		return environment_dict_with_none
	elif keyword == 'agents':
		return environment_dict_with_none[keyword]
	elif keyword == 'hyperparameter':
		return hyperparameter_dict
	elif keyword == 'rl':
		return hyperparameter_dict_rl
	elif keyword == 'sim_market':
		return hyperparameter_dict_sim_market
	elif not keyword:
		return structure_config_dict
	else:
		assert False


def get_structure_with_types_of(top_level: str, second_level: str = None) -> dict:
	"""
	Currently only implemented for 'rl' and 'sim_market'.
	Will return the structure of these configs with the correspondig types.

	Args:
		top_level (str): top level dict key. ('envionment', 'hyperparameter')
		second_level (str, optional): second level dict key. ('rl', 'sim_market', 'agents') Defaults to None.

	Returns:
		dict: with keyword and Django type.
	"""
	assert top_level == 'hyperparameter' and second_level == 'rl' \
		or top_level == 'hyperparameter' and second_level == 'sim_market', \
		f'It is only implemented for "hyperparameter" and "rl, sim_market" not {top_level}, {second_level}'
	if second_level == 'rl':
		possible_attributes = get_all_possible_rl_hyperparameter()
	if second_level == 'sim_market':
		possible_attributes = get_all_possible_sim_market_hyperparameter()
	final_attributes = []
	for attr in possible_attributes:
		final_attributes += [(attr[0], convert_python_type_to_django_type(attr[1]))]
	return final_attributes


def remove_none_values_from_dict(dict_with_none_values: dict) -> dict:
	return {key: value for key, value in dict_with_none_values.items() if value is not None}


def to_config_class_name(name: str) -> str:
	return ''.join([x.title() for x in name.split('_')]) + 'Config'


def to_config_keyword(class_name: str) -> str:
	name_without_config = str(class_name).rsplit('.')[-1][:-2].replace('Config', '').lower()
	# revert the '_'
	return 'sim_market' if name_without_config == 'simmarket' else name_without_config
