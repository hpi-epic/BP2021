import requests

from .api_response import APIResponse
from .models.container import update_container

DOCKER_API = 'http://127.0.0.1:8000'  # remember to include the port and the protocol, i.e. http://


def send_get_request(wanted_action: str, container_id: str) -> APIResponse:
	"""
	Sends a get request to the API with the wanted action for a wanted container.

	Args:
		wanted_action (str): The API call that should be performed. Needs to be a key in `raw_data`
		container_id (str): id of container the action should be performed on

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	try:
		response = requests.get(f'{DOCKER_API}/{wanted_action}', params={'id': str(container_id)})
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
		response = requests.get(f'{DOCKER_API}/{wanted_action}', params={'id': str(wanted_container)}, stream=True)
	except requests.exceptions.RequestException:
		return APIResponse('error', content='The API is unavailable')
	if response.ok:
		return APIResponse('success', content=response)
	return _error_handling_API(response)


def send_post_request(route: str, body: dict, num_experiments: int) -> APIResponse:
	try:
		response = requests.post(f'{DOCKER_API}/{route}', json=body, params={'num_experiments': num_experiments})
	except requests.exceptions.RequestException:
		return APIResponse('error', content='The API is unavailable')
	if response.ok:
		return APIResponse('success', content=response.json())
	return _error_handling_API(response)


def stop_container(container_id: str) -> APIResponse:
	"""
	Sends an API request to stop and remove the container on the remote machine.

	Args:
		post_request (str): id of container that should be stopped

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	response = send_get_request('remove', container_id)
	if response.ok() or response.not_found():
		# mark container as archived
		update_container(container_id, {'health_status': 'archived'})
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
	print(f'Got status code {response.status_code} from API')
	return APIResponse('error', content='Something is wrong with the API. Please try again later.', http_status=response.status_code)
