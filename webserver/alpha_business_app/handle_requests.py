import requests

from .models import update_container

# start api with uvicorn app:app --reload
DOCKER_API = 'http://192.168.159.134:8000'  # http://127.0.0.1:8000'


def send_post_request(route: str, body: dict, command: str) -> dict:
	response = requests.post(DOCKER_API + '/' + route, json=body, params={'command': command})
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


def send_get_request_with_streaming(wanted_action: str, wanted_container: str):
	response = requests.get(DOCKER_API + '/' + wanted_action, params={'id': str(wanted_container)}, stream=True)
	if response.ok:
		return response
	else:
		return None


def stop_container(post_request) -> bool:
	response = send_get_request('remove', post_request)
	if response:
		# mark container as archived
		# TODO add a success message for the user
		update_container(response['id'], {'health_status': 'archived'})
		return True
	return False
