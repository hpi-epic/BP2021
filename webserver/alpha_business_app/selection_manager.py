import re

from django.shortcuts import render

from recommerce.configuration.environment_config import get_class


class SelectionManager:
	def __init__(self) -> None:
		pass

	def get_marketplace_options(self):
		import recommerce.market.circular.circular_sim_market as circular_market
		import recommerce.market.linear.linear_sim_market as linear_market

		# we accept any circular marketplace starting with CircularEconomy or LinearEconomy and at least one more character
		circular_marketplaces = list(set(filter(lambda class_name: re.match('^CircularEconomy..*$', class_name), dir(circular_market))))
		circular_tuples = [(f'recommerce.market.circular.circular_sim_market.{market}', market) for market in sorted(circular_marketplaces)]

		visible_linear_names = list(set(filter(lambda class_name: re.match('^LinearEconomy..*$', class_name), dir(linear_market))))
		linear_tuples = [(f'recommerce.market.linear.linear_sim_market.{market}', market) for market in sorted(visible_linear_names)]

		return circular_tuples + linear_tuples

	def get_agent_options_for_marketplace(self, marketplace_class: str):
		agent_classes = get_class(marketplace_class).get_possible_agents()
		return self._to_tuple_list(agent_classes)

	def _get_number_of_competitors_for_marketplace(self, marketplace_class: str):
		return get_class(marketplace_class).get_num_competitors()

	def get_competitor_options_for_marketplace(self, marketplace_class: str):
		competitor_classes = get_class(marketplace_class).get_competior_classes()
		# print('########################')
		# print(competitor_classes)
		return self._to_tuple_list(competitor_classes)

	def get_correct_agents_html_on_marketplace_change(self, request, marketplace_class: str, raw_agents_html: str) -> str:
		# TODO: find a way to use correct xml parsing
		# get all competitors
		all_competitor_selections = re.findall('<select.*class=.*competitor-agent-class.*>[^</select>]*</select>', raw_agents_html)
		if not all_competitor_selections:
			return raw_agents_html
		# get new competitor options
		competitor_classes = self.get_competitor_options_for_marketplace(marketplace_class)
		new_select = render(request, 'configuration_items/selection_list.html', {'selections': competitor_classes}).content.decode('utf-8')
		# replace current options with new options
		for competitor_selection in all_competitor_selections:
			select_line = competitor_selection.split('\n')
			new_total_html = '\n'.join([select_line[0], new_select, select_line[-1]])
			raw_html = raw_agents_html.replace(competitor_selection, new_total_html)

		return raw_html

	def get_correct_agents_html_on_add_agent(self, request, marketplace_class: str, raw_agents_html: str) -> str:
		pass

	def _to_tuple_list(self, list_of_class_names) -> list:
		final_tuples = []
		for class_name in list_of_class_names:
			raw_parts = str(class_name).rsplit('.', 1)
			visible_name = raw_parts[1]
			actual_class = raw_parts[0] + '.' + visible_name
			final_tuples += [(actual_class, visible_name)]
		return final_tuples
