import copy
import re
from uuid import uuid4

import lxml.html
from django.shortcuts import render

from recommerce.configuration.environment_config import get_class


class SelectionManager:
	def __init__(self) -> None:
		self.current_marketplace = None

	def get_marketplace_options(self):
		import recommerce.market.circular.circular_sim_market as circular_market
		import recommerce.market.linear.linear_sim_market as linear_market

		# we accept any circular marketplace starting with CircularEconomy or LinearEconomy and at least one more character
		circular_marketplaces = list(set(filter(lambda class_name: re.match('^CircularEconomy..*$', class_name), dir(circular_market))))
		circular_tuples = [(f'recommerce.market.circular.circular_sim_market.{market}', market) for market in sorted(circular_marketplaces)]

		visible_linear_names = list(set(filter(lambda class_name: re.match('^LinearEconomy..*$', class_name), dir(linear_market))))
		linear_tuples = [(f'recommerce.market.linear.linear_sim_market.{market}', market) for market in sorted(visible_linear_names)]

		self.current_marketplace = circular_tuples[0][0]
		return circular_tuples + linear_tuples

	def get_agent_options_for_marketplace(self) -> list:
		return self._to_tuple_list(get_class(self.current_marketplace).get_possible_agents())

	def _get_number_of_competitors_for_marketplace(self) -> int:
		return get_class(self.current_marketplace).get_num_competitors()

	def get_competitor_options_for_marketplace(self) -> list:
		return self._to_tuple_list(get_class(self.current_marketplace).get_competitor_classes())

	def get_correct_agents_html_on_marketplace_change(self, request, marketplace_class: str, raw_agents_html: str) -> str:
		# parse string to xml tree
		self.current_marketplace = marketplace_class
		expected_num_competitors = self._get_number_of_competitors_for_marketplace()
		html = lxml.html.fromstring(raw_agents_html)
		collapse_agent_container = list(html)[1]
		big_accordion_body = list(collapse_agent_container)[0]

		# are we in an oligopoly scenario?
		if expected_num_competitors > 5:
			# add add-more-button
			button = html.find_class('add-more')
			if not button:
				add_more_button_html = render(request, 'configuration_items/add_more_button.html').content.decode('utf-8')
				add_more_button = lxml.html.fragment_fromstring(add_more_button_html)
				big_accordion_body.append(add_more_button)
			html = self._update_comp_options(html, request)
			return lxml.html.tostring(html)
		# we are not in oligopoly, so we don't need to add more
		html = self._remove_add_more_button(html, big_accordion_body)
		all_competitor_accordions = html.find_class('accordion-competitor')
		diff = expected_num_competitors - len(all_competitor_accordions)
		if diff < 0:
			# we need to delte the last competitors
			for index in range(diff * -1):
				big_accordion_body.remove(all_competitor_accordions[-(index+1)])
		elif diff > 0:
			# we need to add more competitors
			competitor_html = render(request, 'configuration_items/agent.html', {'id': str(uuid4()), 'name': 'Competitor'}).content.decode('utf-8')
			competitor_element = lxml.html.fragment_fromstring(competitor_html)
			big_accordion_body.append(competitor_element)

		html = self._update_comp_options(html, request)

		return lxml.html.tostring(html)

	def _update_comp_options(self, html, request):
		# find all <selecty>
		competitor_selections = html.find_class('competitor-agent-class')
		if not competitor_selections:
			return html
		# get the new options
		competitor_classes = self.get_competitor_options_for_marketplace()
		new_select = render(request, 'configuration_items/selection_list.html', {'selections': competitor_classes}).content.decode('utf-8')
		new_options = lxml.html.fragments_fromstring(new_select)
		# remove old options and append new options
		for select in competitor_selections:
			old_options = list(select)
			for option in old_options:
				select.remove(option)
			for option in new_options:
				select.append(copy.deepcopy(option))
		return html

	def _remove_add_more_button(self, html, big_accordion_div):
		button = html.find_class('add-more')
		if button:
			big_accordion_div.remove(button[0])
		return html

	def _to_tuple_list(self, list_of_class_names) -> list:
		final_tuples = []
		for class_name in list_of_class_names:
			raw_parts = str(class_name).rsplit('.', 1)
			visible_name = raw_parts[1]
			actual_class = raw_parts[0] + '.' + visible_name
			final_tuples += [(actual_class, visible_name)]
		return final_tuples
