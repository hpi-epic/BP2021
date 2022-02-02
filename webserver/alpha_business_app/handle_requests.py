import requests

from .models import Container

# start api with uvicorn app:app --reload
DOCKER_API = 'http://127.0.0.1:8000'


def send_post_request(route: str, body) -> dict:
	response = requests.post(DOCKER_API + '/' + route, json=body)
	if response.ok:
		return response.json()
	else:
		return None


def send_get_request(wanted_action: str, raw_data) -> dict:
	wanted_container = raw_data[wanted_action]
	response = requests.get(DOCKER_API + '/' + wanted_action, params={'id': str(wanted_container)})
	if response.ok:
		return response.json()
	else:
		return None


def update_container(id: str, updated_values: dict) -> None:
	saved_container = Container.objects.get(container_id=id)
	print(updated_values)
	for key, value in updated_values.items():
		setattr(saved_container, key, value)
	saved_container.save()


def send_get_request_with_streaming(wanted_action: str, wanted_container: str):
	response = requests.get(DOCKER_API + '/' + wanted_action, params={'id': str(wanted_container)}, stream=True)
	if response.ok:
		return response
	else:
		return None