# app.py
import hashlib
import os
import time

import uvicorn
from docker_manager import DockerInfo, DockerManager
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# This file should expose a RESTful api for using the docker container with the following routes:
# POST /start/<command><config>
# GET /health/<id>
# GET /logs/<id><timestamps><stream><tail>
# GET /data/<id><path>
# GET /data/tensorboard/<id>
# GET /remove/<id>

# start API with
# uvicorn app:app --reload
# If using a remote machine use
# uvicorn --host 0.0.0.0 app:app --reload
# instead to expose it to the local network
manager = DockerManager()

app = FastAPI()


def is_invalid_status(status: str) -> bool:
	"""
	Utitlity function that checks a given string against a set of valid container statuses.

	If the string does not match any of them, the status is deemed invalid and the API should return a 404.

	Args:
		status (str): The status to check against the set.

	Returns:
		bool: Whether or not the status indicates a successful operation.
	"""
	valid_container_statuses = {'running', 'paused', 'exited', 'restarting', 'created'}
	# extra check for 'exited (' since we return some exited statuses including the exit code
	return status not in valid_container_statuses and 'exited (' not in status


def verify_token(request: Request) -> bool:
	"""
	verifies for a given request that the header contains the right AUTHORIZATION_TOKEN.
	Warning: This cannot be considered 100% secure, without https, any network sniffer can read the token

	Args:
		request (Request): The request to the API

	Returns:
		bool: if the given authorization token matches our authorization token.
	"""
	try:
		token = request.headers['Authorization']
	except KeyError:
		print('The request did not set an Authorization header')
		return False
	master_secret_as_int = sum(ord(c) for c in os.environ['AUTHORIZATION_TOKEN'])
	current_time = int(time.time() / 3600)  # unix time in hours
	# token, that is currently expected
	expected_this_token = hashlib.sha256(str(master_secret_as_int + current_time).encode('utf-8')).hexdigest()
	# token that was expected last hour
	expected_last_token = hashlib.sha256(str(master_secret_as_int + (current_time - 3600)).encode('utf-8')). hexdigest()
	return token == expected_this_token or token == expected_last_token


@app.post('/start')
async def start_container(num_experiments: int, config: Request, authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	Start a container with the specified config.json and perform a command on it.

	Args:
		num_experiments (int): the number of container, that should be started with this configuration
		config (Request):  The combined hyperparameter_config.json and environment_config_command.json files that should be sent to the container.

	Returns:
		JSONResponse: If starting was successfull the response contains multiple dicts, one for each started container.
			If not, there will be one dict with an error message
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	all_container_infos = manager.start(config=await config.json(), count=num_experiments)
	# check if all prerequisites were met
	if type(all_container_infos) == DockerInfo:
		return JSONResponse(status_code=404, content=vars(all_container_infos))

	return_dict = {}
	for index in range(num_experiments):
		if (is_invalid_status(all_container_infos[index].status) or all_container_infos[index].data is False):
			return JSONResponse(status_code=404, content=vars(all_container_infos[index]))
		return_dict[index] = vars(all_container_infos[index])
	print(f'successfully started {num_experiments} container')
	return JSONResponse(return_dict, status_code=200)


@app.get('/health/')
async def is_container_alive(id: str, authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	Check the status of a container.

	Most other commands also return the status of the container in the `status` field, or in their header.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the status request.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	container_info = manager.health(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info), status_code=200)


@app.get('/logs/')
async def get_container_logs(id: str,
	timestamps: bool = False,
	stream: bool = False,
	tail: int = 'all',
	authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	Get the logs of a container.

	Args:
		id (str): The id of the container.
		timestamps (bool): Whether or not timestamps should be included in the logs. Defaults to False.
		stream (bool): Whether to stream the logs instead of directly retrieving them. Defaults to False.
		tail (int): How many lines at the end of the logs should be returned. int or 'all'. Defaults to 'all'.

	Returns:
		JSONResponse: If stream=False. The response of the log request.
		StreamingResponse: If stream=True. The response of the log request.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	container_info = manager.get_container_logs(id, timestamps, stream, tail)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	elif stream:
		return StreamingResponse(
			container_info.stream,
			headers={
				'Container-ID': f'{container_info.id}',
				'Container-Status': f'{container_info.status}',
			}, status_code=200)
	else:
		return JSONResponse(content=vars(container_info), status_code=200)


@app.get('/data/')
async def get_container_data(id: str, path: str = '/app/results', authorized: bool = Depends(verify_token)) -> StreamingResponse:
	"""
	Extract a folder or file from a container.

	Args:
		id (str): The id of the container.
		path (str, optional): The path of the folder or file that should be extracted. Defaults to '/app/results'.

	Returns:
		StreamingResponse: A stream generator that will download the requested path as a .tar archive.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	container_info = manager.get_container_data(id, path)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return StreamingResponse(
			container_info.stream,
			headers={
				'Content-Disposition': f'filename={container_info.data}.tar',
				'Container-ID': f'{container_info.id}',
				'Container-Status': f'{container_info.status}',
			},
			media_type='application/x-tar', status_code=200)


@app.get('/data/tensorboard/')
async def get_tensorboard_link(id: str, authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	Start a tensorboard session in a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the tensorboard request encapsuled in a DockerInfo JSON. A link is in the data field.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	container_info = manager.start_tensorboard(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info), status_code=200)


@app.get('/pause/')
async def pause_container(id: str, authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	Pause a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the pause request encapsuled in a DockerInfo JSON.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	container_info = manager.pause(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info), status_code=200)


@app.get('/unpause/')
async def unpause_container(id: str, authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	Unpause a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the unpause request encapsuled in a DockerInfo JSON.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	container_info = manager.unpause(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info), status_code=200)


@app.get('/remove/')
async def remove_container(id: str, authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	Stop and remove a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the remove request encapsuled in a DockerInfo JSON. Status will be 'removed' if successful.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	container_info = manager.remove_container(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info), status_code=200)


@app.get('/api_health')
async def check_if_api_is_available(authorized: bool = Depends(verify_token)) -> JSONResponse:
	"""
	This is a route you can call to see if the API is available.
	If the API is unavailable, this will of course not actually get called which the Webserver will catch.
	But if it is available, we also check if docker is responsive, i.e. if `manager._client` is a valid docker.DockerClient.

	Returns:
		JSONResponse: A json with a `status` field and status code indicating if docker is available.
	"""
	if not authorized:
		return JSONResponse(status_code=401, content=vars(DockerInfo('', 'Not authorized')))
	docker_status = manager.ping()
	status_code = 200 if docker_status else 404
	return JSONResponse({'status': docker_status}, status_code=status_code)


if __name__ == '__main__':
	uvicorn.run('app:app',
		host='0.0.0.0',
		port=8000,
		ssl_keyfile='/etc/sslzertifikat/api_cert.key',
		ssl_certfile='/etc/sslzertifikat/api_cert.crt')
