import recommerce.market.circular.circular_sim_market as circular_market
import recommerce.market.linear.linear_sim_market as linear_market
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
