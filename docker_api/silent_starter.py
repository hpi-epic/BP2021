import datetime
import hashlib
import json
import logging
import os
import time

import requests
from utils import setup_logging

DOCKER_API = 'https://vm-midea03.eaalab.hpi.uni-potsdam.de:8000'  # remember to include the port and the protocol, i.e. http://

setup_logging('silent_starter')


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
			master_secret = os.environ['API_TOKEN']
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
		'verify': os.path.join('..', 'webserver', 'ssl_certificates', 'api.crt')
	}


def send_post_request(route: str, body: dict, params: dict):
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
		logging.info('could not send post request')
		return None
	if response.ok:
		return response.json()
	logging.info('API response not okay')
	return None


def stop_container(container_id: str):
	"""
	Sends an API request to stop and remove the container on the remote machine.

	Args:
		post_request (str): id of container that should be stopped

	Returns:
		APIResponse: Response from the API converted into our special format.
	"""
	response = send_get_request('remove', container_id)
	if response.ok() or response.not_found():
		return True
	return False


def send_get_request(wanted_action: str, container_id: str):
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
		logging.info('Could not send get request to api')
		return None
	if response.ok:
		return response.json()
	logging.info('The api response was not okay')
	return None


def is_time_between(begin_time, end_time, check_time=None):
	check_time = check_time or datetime.datetime.now().time()
	if begin_time < end_time:
		return check_time >= begin_time and check_time <= end_time
	else:  # midnight
		return check_time >= begin_time or check_time <= end_time


num_container = [1, 2, 3, 4, 5, 6, 7, 8, 9]

with open('./test_config.json', 'r') as file:
	config_dict = json.load(file)

index = 0
last_container_are_stopped = True
running_container = []
first_time = True
while True:
	if is_time_between(datetime.time(8, 0), datetime.time(14, 0)):
		if first_time:
			logging.info('started working')
			first_time = False
		if last_container_are_stopped:
			print(f'sending post request for {index % len(num_container)} container.')
			response = send_post_request('start', config_dict, {'num_experiments': num_container[index % len(num_container)]})
			print(f'got response, and response is: {response is not None}')
			if response:
				for _, c in response.items():
					print(c['id'])
					running_container += [c['id']]
			last_container_are_stopped = False
			print('------------------')
		else:
			stati = []
			print('checking status for container')
			for c in running_container:
				response = send_get_request('health', c)
				stati += [response['status']]
			if any('running' in status for status in stati):
				print('some container are still running')
			else:
				# all are exited, we can start over
				print('stopping container')
				for c in running_container:
					response = stop_container(c)
					if not response:
						logging.warn(f'Could not stop container {c}')
				last_container_are_stopped = True
				index += 1
	else:
		if not first_time:
			logging.info('done working')
		print(f'Not working, it is {datetime.datetime.now().time()}')
		first_time = True
	print('*', running_container)
	if running_container:
		print(send_get_request('logs', running_container[0]))
	time.sleep(305)
