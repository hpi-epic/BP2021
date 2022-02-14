import requests

from .api_response import APIResponse
from .constants import DOCKER_API
from .models import update_container


def send_get_request(wanted_action: str, raw_data: dict) -> APIResponse:
	"""
	Sends a get request to the API with the wanted action for a wanted container.

	Args:
		wanted_action (str): The API call that should be performed. Needs to be a key in `raw_data`
		raw_data (dict): various post parameters,
			must include the wanted action as key and the container_id as the value for this key.

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	wanted_container = raw_data[wanted_action]
	try:
		response = requests.get(DOCKER_API + '/' + wanted_action, params={'id': str(wanted_container)})
	except requests.exceptions.RequestException:
		return APIResponse('error', content='The API is unavailable')
	if response.ok:
		return APIResponse('success', content=response.json())
	return _error_handling_API(response)


def send_get_request_with_streaming(wanted_action: str, wanted_container: str) -> APIResponse:
	"""
	Sends a get request to the API and gets an HttpStreamingResponse as an answer.

	Args:
		wanted_action (str): The API call that should be performed.
		wanted_container (str): The container that should be passed as parameter to the API.

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	try:
		response = requests.get(DOCKER_API + '/' + wanted_action, params={'id': str(wanted_container)}, stream=True)
	except requests.exceptions.RequestException:
		return APIResponse('error', content='The API is unavailable')
	if response.ok:
		return APIResponse('success', content=response)
	return _error_handling_API(response)


def send_post_request(route: str, body: dict, command: str) -> APIResponse:
	"""
	Sends a post request to the API with the requested parameter, a body and a command as parameter

	Args:
		route (str): A post route from the API.
		body (dict): The body that should be send to the API.
		command (str): A command parameter for the API, currently these are `training, monitoring, exampleprinter`

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	try:
		response = requests.post(DOCKER_API + '/' + route, json=body, params={'command': command})
	except requests.exceptions.RequestException:
		return APIResponse('error', content='The API is unavailable')
	if response.ok:
		return APIResponse('success', content=response.json())
	return _error_handling_API(response)


def stop_container(post_request: dict) -> APIResponse:
	"""
	Sends an API request to stop and remove the container on the remote machine.

	Args:
		post_request (dict): parameters for the request, must include the key 'remove' and as its value the container_id

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	response = send_get_request('remove', post_request)
	if response.ok() or response.not_found():
		# mark container as archived
		update_container(post_request['remove'], {'health_status': 'archived'})
		return APIResponse('success', content='You successfully stopped the container')
	return response


def _error_handling_API(response) -> APIResponse:
	"""
	Defines error codes and appropriate response messages if we get error codes from the API.

	Args:
		response (HttpResponse): Raw response from the API

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	if response.status_code < 500:
		return APIResponse('error', content=response.json()['status'], http_status=response.status_code)
	else:
		print('Got status code %s from API' % response.status_code)
		return APIResponse('error', content='something is wrong with the API. Please try again later', http_status=response.status_code)
