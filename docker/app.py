# app.py

from docker_manager import DockerManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

# This file should expose a RESTful api for using the docker container with the following routes:
# POST /start/<docker_id>
# GET /health/<docker_id>
# GET /data/<docker_id>
# GET /data/tensorboard/<docker_id>
# GET /kill/<docker_id>

# before first use on a new machine/ with changes to the environment or the src folder,
# please call run the docker_manager.py file. It initializes the image and takes ages.
# start API with uvicorn app:app --reload
# If using a remote machine use "uvicorn --host 0.0.0.0 app:app --reload" instead to expose it to the local network
manager = DockerManager()

app = FastAPI()


# works
@app.post('/start')
async def start_container(config: Request):
	container_info = manager.start(await config.json())
	return JSONResponse(vars(container_info))


# works
@app.get('/health/')
async def is_container_alive(id: str):
	container_info = manager.health(id)
	return JSONResponse(vars(container_info))


# does not work
@app.get('/data/')
async def get_container_data(id: str):
	container_info = manager.get_container_data(id)
	return JSONResponse(vars(container_info))


# works
@app.get('/exec/')
async def execute_command(id: str, command: str):
	container_info = manager.execute_command(id, command)
	return StreamingResponse(vars(container_info)['stream'])


# works
@app.get('/stop/')
async def stop_container(id: str):
	container_info = manager.stop_container(id)
	return JSONResponse(vars(container_info))


# works
@app.get('/data/tensorboard/')
async def get_tensorboard_link(id: str):
	tb_link = manager.start_tensorboard(id)
	return RedirectResponse(vars(tb_link)['data'])


# works, returns 'removed' and stops the container if it still runs
@app.get('/remove/')
async def remove_container(id: str):
	container_info = manager.remove_container(id)
	return JSONResponse(vars(container_info))


@app.get('/kill/')
async def kill_container(id: str):
	# deprecated in favor of remove
	container_info = manager.remove_container(id)
	return JSONResponse(vars(container_info))
