import copy

import names

from .config_parser import ConfigModelParser
from .models.container import Container


def parse_response_to_database(api_response, config_dict: dict, given_name: str) -> None:
	# get constants
	started_container = api_response.content
	num_experiments = len(started_container)
	name = names.get_first_name() if not given_name else given_name

	# save the used config
	config_object = ConfigModelParser().parse_config(copy.deepcopy(config_dict))
	config_object.name = f'Config for {name}'
	config_object.save()

	command = config_object.environment.task

	for container_count, container_info in started_container.items():
		# check if a container with the same id already exists
		if Container.objects.filter(id=container_info['id']).exists():
			# we will kindly ask the user to try it again and stop the container
			# TODO insert better handling here
			return False, ['error', 'The new container has the same id as an already existing container, please try again.']
		# get name for container
		if num_experiments != 1:
			current_container_name = name + f' ({container_count})'
		else:
			current_container_name = name

		# create the container
		Container.objects.create(id=container_info['id'],
			command=command,
			config=config_object,
			health_status=container_info['status'],
			name=current_container_name)
	return True, name
