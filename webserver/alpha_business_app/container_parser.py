import copy

from .config_parser import ConfigModelParser
from .models.container import Container


def parse_response_to_database(api_response, config_dict: dict, name: str) -> None:
	config_object = ConfigModelParser().parse_config(copy.deepcopy(config_dict))
	config_object.name = f'Config for {name}'
	config_object.save()

	command = config_object.environment.task
	started_container = api_response.content

	for container_count, container_info in started_container.items():
		# check if a container with the same id already exists
		if Container.objects.filter(id=container_info['id']).exists():
			# we will kindly ask the user to try it again and stop the container
			# TODO insert better handling here
			return False, ['error', 'The new container has the same id as an already existing container, please try again.']
		if not name:
			current_container_name = container_info['id'][:10]
		else:
			current_container_name = name + f' ({container_count})'
		Container.objects.create(id=container_info['id'],
			command=command,
			config=config_object,
			health_status=container_info['status'],
			name=current_container_name)
	return True, 'error'
