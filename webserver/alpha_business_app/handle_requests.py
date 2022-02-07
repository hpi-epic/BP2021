import requests

from .api_response import APIResponse
from .constants import DOCKER_API
from .models import update_container


def send_post_request(route: str, body: dict, command: str) -> dict:
	try:
		response = requests.post(DOCKER_API + '/' + route, json=body, params={'command': command})
	except requests.exceptions.RequestException:
		return APIResponse('error', string_response='The API is unavailable')
	if response.ok:
		return APIResponse('success', json_response=response.json())
	return _error_handling_API(response)


def send_get_request(wanted_action: str, raw_data) -> dict:
	wanted_container = raw_data[wanted_action]
	try:
		response = requests.get(DOCKER_API + '/' + wanted_action, params={'id': str(wanted_container)})
	except requests.exceptions.RequestException:
		return APIResponse('error', string_response='The API is unavailable')
	if response.ok:
		return APIResponse('success', json_response=response.json())
	return _error_handling_API(response)


def send_get_request_with_streaming(wanted_action: str, wanted_container: str):
	try:
		response = requests.get(DOCKER_API + '/' + wanted_action, params={'id': str(wanted_container)}, stream=True)
	except requests.exceptions.RequestException:
		return APIResponse('error', string_response='The API is unavailable')
	if response.ok:
		return APIResponse('success', raw_response=response)
	return _error_handling_API(response)


def stop_container(post_request) -> bool:
	response = send_get_request('remove', post_request)
	print(response.status())
	if response.ok() or response.not_found():
		# mark container as archived
		update_container(post_request['remove'], {'health_status': 'archived'})
		return APIResponse('success', string_response='You successfully stopped the container')
	return APIResponse('error', string_response='The container could not be stopped')


def _error_handling_API(response) -> APIResponse:
	if response.status_code < 500:
		return APIResponse('error', string_response=response.json()['status'], http_status=response.status_code)
	else:
		print('Got status code', response.status_code, 'from API')
		return APIResponse('error', string_response='something is wrong with the API. Please try again later', http_status=response.status_code)
