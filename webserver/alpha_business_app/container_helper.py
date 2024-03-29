import copy

import names

from .config_parser import ConfigModelParser
from .models.container import Container


def parse_response_to_database(api_response, config_dict: dict, given_name: str, user) -> None:
	"""
	Parses an API response containing multiple container to the database.

	Args:
		api_response (APIResponse): The converted response from the docker API.
		config_dict (dict): The dict the container have been started with.
		given_name (str): the name the user put into the field
		user (User): the user who did the request
	"""
	started_container = api_response.content
	# check if the api response is correct
	for _, container_info in started_container.items():
		if type(container_info) != dict:
			return False, [], 'The API answer was wrong, please try'

	num_experiments = len(started_container)
	name = names.get_first_name() if not given_name else str(given_name)

	# save the used config
	config_object = ConfigModelParser().parse_config(copy.deepcopy(config_dict))
	config_object.name = f'Config for {name}'
	config_object.user = user
	config_object.save()

	command = config_object.environment.task

	for container_count, container_info in started_container.items():
		# check if a container with the same id already exists
		if Container.objects.filter(id=container_info['id']).exists():
			# we will kindly ask the user to try it again and stop the container
			# TODO insert better handling here
			all_container_ids = [info['id'] for _, info in started_container.items()]
			return False, all_container_ids, 'The new container has the same id as an already existing container, please try again.'
		# get name for container
		if num_experiments != 1:
			current_container_name = str(name) + f' ({str(container_count)})'
		else:
			current_container_name = name

		# create the container
		Container.objects.create(id=container_info['id'],
			command=command,
			config=config_object,
			health_status=container_info['status'],
			name=current_container_name,
			user=user)
	return True, [], name
