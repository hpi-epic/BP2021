# app.py

from docker_manager import DockerManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# This file should expose a RESTful api for using the docker container with the following routes:
# POST /start/<docker_id>
# GET /health/<docker_id>
# GET /data/<docker_id>
# GET /data/tensorboard/<docker_id>
# GET /remove/<docker_id>

# before first use on a new machine/ with changes to the environment or the src folder,
# please call run the docker_manager.py file. It initializes the image and takes ages.
# start API with uvicorn app:app --reload
# If using a remote machine use "uvicorn --host 0.0.0.0 app:app --reload" instead to expose it to the local network
manager = DockerManager()

app = FastAPI()


@app.post('/start')
async def start_container(command: str, config: Request) -> JSONResponse:
	"""
	Start a container with the specified config.json and perform a command on it.
	TODO: The command should be contained in a json-file.

	Args:
		command (str): The key of the command that is to be executed.
		config (Request): The config.json file that should be sent to the container.

	Returns:
		StreamingResponse: The response of the Docker start request. Contains custom header keys for id and status of the container.
	"""
	container_info = await manager.start(config=await config.json(), command=command)
	return StreamingResponse(
		container_info.stream,
		headers={
			'Container-ID': f'{container_info.id}',
			'Container-Status': f'{container_info.status}',
		},
		media_type='application/x-tar')


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
	tb_link = await manager.start_tensorboard(id)
	return JSONResponse(vars(tb_link))


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
	if container_info.status.__contains__('Container not found'):
		return JSONResponse(status_code=404, content=vars(container_info))
	else:
		return JSONResponse(vars(container_info))


# Route kept, but shouldn't be used anymore due to security concerns
# @app.get('/exec/')
# async def execute_command(id: str, command: str) -> StreamingResponse:
# 	"""
# 	Execute a command in a container.

# 	Args:
# 		id (str): The id of the container.
# 		command (str): The key of the command to execute, which will be mapped to the python command internally if it is valid.

# 	Returns:
# 		StreamingResponse: A stream generator that will return the stdout the container produces from the command.
# 	"""
# 	container_info = await manager.execute_command(id, command)
# 	# if container_info.status.__contains__('not allowed'):
# 	# 	return JSONResponse(status_code=404, content=vars(container_info))
# 	# return JSONResponse(vars(container_info))
# 	return StreamingResponse(container_info.stream)


# Route kept, but functionality should be integrated with /remove
# @app.get('/stop/')
# async def stop_container(id: str) -> JSONResponse:
# 	"""
# 	Stop the container. Does not remove it.

# 	Args:
# 		id (str): The id of the container.

# 	Returns:
# 		JSONResponse: The response of the stop request encapsuled in a DockerInfo JSON. Status will be 'stopped' if successful.
# 	"""
# 	container_info = manager._stop_container(id)
# 	if container_info.status.__contains__('Container not found'):
# 		return JSONResponse(status_code=404, content=vars(container_info))
# 	else:
# 		return JSONResponse(vars(container_info))


# Route kept, but shouldn't be necessary
# @app.post('/upload')
# async def upload_config(id: str, config: Request) -> JSONResponse:
# 	"""
# 	Upload a config.json to the container.

# 	Args:
# 		id (str): The id of the container.
# 		config (Request): The config.json that should be uploaded.

# 	Returns:
# 		JSONResponse: The response of the upload request.
# 	"""
# 	container_info = manager._upload_config(id, await config.json())
# 	if container_info.status.__contains__('Container not found'):
# 		return JSONResponse(status_code=404, content=vars(container_info))
# 	else:
# 		return JSONResponse(vars(container_info))
