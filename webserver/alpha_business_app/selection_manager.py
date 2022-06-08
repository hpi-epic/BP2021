import copy
from uuid import uuid4

import lxml.html
from django.shortcuts import render

from recommerce.configuration.utils import get_class

from .utils import get_recommerce_agents_for_marketplace, get_recommerce_marketplaces


class SelectionManager:
	def __init__(self) -> None:
		self.current_marketplace = None

	def get_agent_options_for_marketplace(self) -> list:
		return self._to_tuple_list(get_recommerce_agents_for_marketplace(self.current_marketplace))

	def get_competitor_options_for_marketplace(self) -> list:
		return self._to_tuple_list(self.current_marketplace.get_competitor_classes())

	def get_correct_agents_html_on_marketplace_change(self, request, marketplace_class: str, raw_agents_html: str) -> str:
		"""
		Converts the agent accordion html to an agent accordion, that contains the right agent options.
		It parses the html to xml elements and adds the right options to competitor class selection.

		Args:
			request (HttpRequest): the request made to the server
			marketplace_class (str): marketplace as string in the form 'recommerce...'
			raw_agents_html (str): original html of the agents accordion

		Returns:
			str: correct agent accordion
		"""
		# parse string to xml tree
		html = lxml.html.fromstring(raw_agents_html)

		# get necessary constants
		self.current_marketplace = get_class(marketplace_class)
		expected_num_competitors = self._get_number_of_competitors_for_marketplace()
		collapse_agent_container = list(html)[1]
		big_accordion_body = list(collapse_agent_container)[0]

		if 'Oligopoly' in marketplace_class:
			# add the add-more-button
			button = html.find_class('add-more')
			if not button:
				add_more_button_html = render(request, 'configuration_items/add_more_button.html').content.decode('utf-8')
				add_more_button = lxml.html.fragment_fromstring(add_more_button_html)
				big_accordion_body.append(add_more_button)
			html = self._update_comp_options(html, request)
			return lxml.html.tostring(html)

		# we are not in oligopoly, so we don't need the add-more-button
		html = self._remove_add_more_button(html, big_accordion_body)
		all_competitor_accordions = html.find_class('accordion-competitor')
		# figure out if we have to remove or add competitor accordions
		diff = expected_num_competitors - len(all_competitor_accordions)
		if diff < 0:
			# we need to remove the last competitors
			for index in range(diff * -1):
				big_accordion_body.remove(all_competitor_accordions[-(index + 1)])
		elif diff > 0:
			# we need to add more competitors
			competitor_html = render(request, 'configuration_items/agent.html', {'id': str(uuid4()), 'name': 'Competitor'}).content.decode('utf-8')
			competitor_element = lxml.html.fragment_fromstring(competitor_html)
			big_accordion_body.append(competitor_element)

		html = self._update_comp_options(html, request)

		return lxml.html.tostring(html)

	def get_marketplace_options(self) -> list:
		recommerce_marketplaces = get_recommerce_marketplaces()
		self.current_marketplace = get_class(recommerce_marketplaces[0])
		return self._to_tuple_list(recommerce_marketplaces)

	def _get_task_options(self) -> list:
		return [
			('training', 'training', 'starts a training session'),
			('agent_monitoring', 'agent_monitoring', 'monitors the performance of the agents in a marketplace'),
			('exampleprinter', 'exampleprinter', 'one episode of the agent, can be monitored using tensorboard')
		]

	def _get_number_of_competitors_for_marketplace(self) -> int:
		return self.current_marketplace.get_num_competitors()

	def _get_class_description(self, class_str: str) -> str:
		return get_class(class_str).__doc__

	def _remove_add_more_button(self, html, big_accordion_div):
		button = html.find_class('add-more')
		if button:
			big_accordion_div.remove(button[0])
		return html

	def _to_tuple_list(self, list_of_class_names: list) -> list:
		"""
		Returns a list of tuples needed for the html `select` statement. The docstring of the class will be used as hovertext.

		Args:
			list_of_class_names (list): list of str containing recommerce class names in the form 'recommerce...'

		Returns:
			list: of tuples of strings on the form:
				(actual class starting with 'recommerce...', name that should be shown as option, hovertext for selection)
		"""
		final_tuples = []
		for class_name in list_of_class_names:
			raw_parts = str(class_name).rsplit('.', 1)
			visible_name = raw_parts[1]
			actual_class = raw_parts[0] + '.' + visible_name
			final_tuples += [(actual_class, visible_name, self._get_class_description(actual_class))]
		return final_tuples

	def _update_comp_options(self, html, request) -> lxml.html.HtmlElement:
		"""
		Removes all children of html element with class 'competitor-agent-class' and adds new selection options to it

		Args:
			html (lxml.html.HtmlElement): parsed html as xml tree
			request (HttpRequest): request made to the server

		Returns:
			lxml.html.HtmlElement: the updated html
		"""
		# find all <select> with class 'competitor-agent-class'
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
