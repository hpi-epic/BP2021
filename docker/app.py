# app.py

import uvicorn
from docker_manager import DockerManager
from fastapi import FastAPI, Request
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


@app.post('/start')
async def start_container(config: Request) -> JSONResponse:
	"""
	Start a container with the specified config.json and perform a command on it.

	Args:
		config (Request): The combined hyperparameter_config.json and environment_config_command.json files that should be sent to the container.

	Returns:
		JSONResponse: The response of the Docker start request. Contains the port used on the host in the data-field.
	"""
	container_info = manager.start(config=await config.json())
	if (is_invalid_status(container_info.status) or container_info.data is False):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))


@app.get('/health/')
async def is_container_alive(id: str) -> JSONResponse:
	"""
	Check the status of a container.

	Most other commands also return the status of the container in the `status` field, or in their header.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the status request.
	"""
	container_info = manager.health(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))


@app.get('/logs/')
async def get_container_logs(id: str, timestamps: bool = False, stream: bool = False, tail: int = 'all') -> JSONResponse:
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
	container_info = manager.get_container_logs(id, timestamps, stream, tail)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	elif stream:
		return StreamingResponse(
			container_info.stream,
			headers={
				'Container-ID': f'{container_info.id}',
				'Container-Status': f'{container_info.status}',
			})
	else:
		return JSONResponse(content=vars(container_info))


@app.get('/data/')
async def get_container_data(id: str, path: str = '/app/results') -> StreamingResponse:
	"""
	Extract a folder or file from a container.

	Args:
		id (str): The id of the container.
		path (str, optional): The path of the folder or file that should be extracted. Defaults to '/app/results'.

	Returns:
		StreamingResponse: A stream generator that will download the requested path as a .tar archive.
	"""
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
			media_type='application/x-tar')


@app.get('/data/tensorboard/')
async def get_tensorboard_link(id: str) -> JSONResponse:
	"""
	Start a tensorboard session in a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the tensorboard request encapsuled in a DockerInfo JSON. A link is in the data field.
	"""
	container_info = manager.start_tensorboard(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))


@app.get('/pause/')
async def pause_container(id: str) -> JSONResponse:
	"""
	Pause a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the pause request encapsuled in a DockerInfo JSON.
	"""
	container_info = manager.pause(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))


@app.get('/unpause/')
async def unpause_container(id: str) -> JSONResponse:
	"""
	Unpause a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the unpause request encapsuled in a DockerInfo JSON.
	"""
	container_info = manager.unpause(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))


@app.get('/remove/')
async def remove_container(id: str) -> JSONResponse:
	"""
	Stop and remove a container.

	Args:
		id (str): The id of the container.

	Returns:
		JSONResponse: The response of the remove request encapsuled in a DockerInfo JSON. Status will be 'removed' if successful.
	"""
	container_info = manager.remove_container(id)
	if is_invalid_status(container_info.status):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))


@app.get('/api_health')
async def check_if_api_is_available() -> JSONResponse:
	"""
	This is a route you can call to see if the API is available.
	If the API is unavailable, this will of course not actually get called which the Webserver will catch.
	But if it is available, we also check if docker is responsive, i.e. if `manager._client` is a valid docker.DockerClient.

	Returns:
		JSONResponse: A json with a `status` field and status code indicating if docker is available.
	"""
	docker_status = manager.ping()
	status_code = 200 if docker_status else 404
	return JSONResponse({'status': docker_status}, status_code=status_code)


if __name__ == '__main__':
	uvicorn.run('app:app', host='0.0.0.0', port=8000)
