import datetime
import hashlib
import os
import time

import requests

from .api_response import APIResponse
from .models.container import update_container

DOCKER_API = 'https://vm-midea03.eaalab.hpi.uni-potsdam.de:8000'  # remember to include the port and the protocol, i.e. http://
# DOCKER_API = 'http://localhost:8000'


def _get_api_token() -> str:
	"""
	Returns a time limited API token for the API.

	Returns:
		str: API token to be sent to the API.
	"""
	try:
		with open('./.env.txt', 'r') as file:
			lines = file.readlines()
			master_secret = lines[1].strip()
	except FileNotFoundError:
		print('No .env file found, using environment variable instead.')
		try:
			master_secret = os.environ['API_TOKEN'].strip()
		except KeyError:
			print('Could not get API key')
			return 'abc'
	master_secret_as_int = sum(ord(c) for c in master_secret)
	current_time = int(time.time() / 3600)  # unix time in hours
	return hashlib.sha256(str(master_secret_as_int + current_time).encode('utf-8')).hexdigest()


def _default_request_parameter(wanted_action: str, params: dict) -> dict:
	"""
	All parameters that are the same for all requests.

	Args:
		wanted_action (str): api route
		params (dict): parameter for the request

	Returns:
		dict: default parameters as dictionary
	"""
	return {
		'url': f'{DOCKER_API}/{wanted_action}',
		'params': params,
		'headers': {'Authorization': _get_api_token()},
		'verify': os.path.join('.', 'ssl_certificates', 'api.crt')
	}


def send_get_request(wanted_action: str, container_id: str) -> APIResponse:
	"""
	Sends a get request to the API with the wanted action for a wanted container.

	Args:
		wanted_action (str): The API call that should be performed.
		container_id (str): id of container the action should be performed on

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	try:
		response = requests.get(**_default_request_parameter(wanted_action, {'id': str(container_id)}))
	except requests.exceptions.RequestException:
		return APIResponse('error', content='The API is unavailable')
	if response.ok:
		return APIResponse('success', content=response.json())
	return _error_handling_API(response)


def send_get_request_with_streaming(wanted_action: str, container_id: str) -> APIResponse:
	"""
	Sends a get request to the API and gets an HttpStreamingResponse as an answer.

	Args:
		wanted_action (str): The API call that should be performed.
		container_id (str): id of container the action should be performed on.

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	try:
		response = requests.get(**_default_request_parameter(wanted_action, {'id': str(container_id)}), stream=True)
	except requests.exceptions.RequestException:
		return APIResponse('error', content='The API is unavailable')
	if response.ok:
		return APIResponse('success', content=response)
	return _error_handling_API(response)


def send_post_request(route: str, body: dict, params: dict) -> APIResponse:
	"""
	Sends a post request to the API on the specific rout with a json as body and parameters for the post request

	Args:
		route (str): route for the post to the API.
		body (dict): dict that will be send in the body of the request as json
		params (dict): other parameter for this operation

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	try:
		response = requests.post(**_default_request_parameter(route, params), json=body)
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


def get_api_status() -> dict:
	"""
	Checks if the API is available and returns the parameters the template should be rendered with

	Returns:
		dict: parameter for the api button template.
	"""
	try:
		api_is_available = requests.get(**_default_request_parameter('api_health', None), timeout=1)
	except requests.exceptions.RequestException:
		current_time = datetime.datetime.now().strftime('%H:%M:%S')
		return {'api_timeout': f'API unavailable - {current_time}'}

	current_time = datetime.datetime.now().strftime('%H:%M:%S')
	if api_is_available.status_code == 200:
		return {'api_success': f'API available - {current_time}'}
	if api_is_available.status_code == 401:
		return {}
	return {'api_docker_timeout': f'Docker unavailable - {current_time}'}


def websocket_url() -> str:
	return 'wss://vm-midea03.eaalab.hpi.uni-potsdam.de:8001/wss'


def _error_handling_API(response) -> APIResponse:
	"""
	Defines error codes and appropriate response messages if we get error codes from the API.

	Args:
		response (HttpResponse): Raw response from the API

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	try:
		if response.status_code < 500:
			return APIResponse('error', content=response.json()['status'], http_status=response.status_code)
		print(f'Got status code {response.status_code} from API')
		return APIResponse('error', content='Something is wrong with the API. Please try again later.', http_status=response.status_code)
	except Exception:
		return APIResponse('error', content='The API response is invalid')
