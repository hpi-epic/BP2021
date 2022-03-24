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
	if (container_info.status.__contains__('Command not allowed')
		or container_info.status.__contains__('Image not found')
		or container_info.status.__contains__('The config is missing')  # missing hyperparameter or environment field
		or container_info.data is False):
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
	if container_info.status.__contains__('Container not found'):
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
	if container_info.status.__contains__('Container not found'):
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
	if (container_info.status.__contains__('Container not found')
		or container_info.status.__contains__('The requested path does not exist on the container')):
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
	if container_info.status.__contains__('Container not found') or container_info.status.__contains__('Container is not running'):
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
	if container_info.status.__contains__('Container not found') or container_info.status.__contains__('Container not paused successfully'):
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
	if container_info.status.__contains__('Container not found') or container_info.status.__contains__('Container not unpaused successfully'):
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
	if container_info.status.__contains__('Container not found') or container_info.status.__contains__('Container not stopped successfully'):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))

if __name__ == '__main__':
	uvicorn.run('app:app', host='0.0.0.0', port=8000)
