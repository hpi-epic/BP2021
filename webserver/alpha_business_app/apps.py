from django.apps import AppConfig

from recommerce.configuration.utils import get_class

from .utils import get_recommerce_agents_for_marketplace, get_recommerce_marketplaces

# from django.db import models


class AlphaBusinessAppConfig(AppConfig):
	"""
	This is our app.
	"""
	name = 'alpha_business_app'

	def ready(self) -> None:
		"""
		This code is executed on startup of our app.
		We use it to define the RL Config model.
		"""
		print('**********************************')
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
		print(set(all_attributes))

		# attrs = {
		# 	'name': models.CharField(max_length=32),
		# 	'__module__': 'alpha_business_app.models'
		# }
		# Animal = type('Animal', (models.Model,), attrs)
		return super().ready()
