import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
from recommerce.configuration.environment_config import EnvironmentConfig
from recommerce.configuration.hyperparameter_config import HyperparameterConfigValidator
from recommerce.configuration.utils import get_class


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


def get_agent_hyperparameter(agent: str) -> list:
	agent_class = get_class(agent)
	agent_specs = agent_class.get_configurable_fields()
	# name, input_type, error
	all_parameter = []
	for spec in agent_specs:
		this_parameter = {}
		this_parameter['name'] = spec[0]
		this_parameter['input_type'] = _convert_python_type_to_input_type(spec[1])
		all_parameter += [this_parameter]
	return all_parameter


def _convert_python_type_to_input_type(to_convert) -> str:
	return 'number' if to_convert == float or to_convert == int else 'text'


def convert_python_type_to_django_type(to_convert) -> str:
	from django.db import models
	if to_convert == float:
		return models.FloatField
	elif to_convert == int:
		return models.IntegerField
	else:
		return models.CharField


def get_all_possible_rl_hyperparameter():
	all_marketplaces = get_recommerce_marketplaces()
	all_agents = []
	for marketplace_str in all_marketplaces:
		marketplace = get_class(marketplace_str)
		all_agents += get_recommerce_agents_for_marketplace(marketplace)

	all_attributes = []
	for agent_str in all_agents:
		agent = get_class(agent_str)
		try:
			all_attributes += agent.get_configurable_fields()
		except NotImplementedError:
			print(f'please check the installation of the recommerce package! Agent: {agent} does not implement get_configurable_fields')
	return set(all_attributes)


def capitalize(word: str) -> str:
	return word.upper() if len(word) <= 1 else word[0].upper() + word[1:]


def to_config_class_name(name: str) -> str:
	return ''.join([capitalize(x) for x in name.split('_')]) + 'Config'


def to_config_keyword(class_name: str) -> str:
	name_without_config = str(class_name).rsplit('.')[-1][:-2].replace('Config', '').lower()
	# revert the '_'
	return 'sim_market' if name_without_config == 'simmarket' else name_without_config


def remove_none_values_from_dict(dict_with_none_values: dict) -> dict:
	return {key: value for key, value in dict_with_none_values.items() if value is not None}


def get_structure_dict_for(keyword) -> dict:
	assert keyword in ['environment', 'hyperparameter', 'sim_market', 'rl', 'agents', ''], f'Your keyword {keyword} is not recognized.'
	environment_dict = EnvironmentConfig.get_required_fields('top-dict')
	environment_dict_with_none = {key: None for key in environment_dict.keys()}
	environment_dict_with_none['agents'] = []

	hyperparameter_dict_sim_market = HyperparameterConfigValidator.get_required_fields('sim_market')
	hyperparameter_dict_sim_market_with_none = {key: None for key in hyperparameter_dict_sim_market.keys()}

	hyperparameter_dict_rl = {parameter[0]: None for parameter in get_all_possible_rl_hyperparameter()}

	hyperparameter_dict = {
		'sim_market': hyperparameter_dict_sim_market_with_none,
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
		return hyperparameter_dict_sim_market_with_none
	elif not keyword:
		return structure_config_dict
	else:
		assert False


def get_structure_with_types_of(top_level: str, second_level: str = None) -> dict:
	assert top_level == 'hyperparameter' and second_level == 'rl', \
		f'It is only implemented for "hyperparameter" and "rl" not {top_level}, {second_level}'
	possible_attributes = get_all_possible_rl_hyperparameter()
	final_attributes = []
	for attr in possible_attributes:
		final_attributes += [(attr[0], convert_python_type_to_django_type(attr[1])(default=None))]
	return final_attributes
