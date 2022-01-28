# app.py
from docker_manager import DockerManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# This file should expose a RESTful api for using the docker container with the following routes:
# POST /start/<docker_id>
# GET /health/<docker_id>
# GET /data/<docker_id>
# GET /data/tensorboard/<docker_id>
# GET /kill/<docker_id>

# start API with uvicorn app:app --reload
# If using a remote machine use "uvicorn --host 0.0.0.0 app:app --reload" instead to expose it to the local network
manager = DockerManager()

app = FastAPI()


@app.post('/start')
async def start_container(config: Request):
	container_info = manager.start(await config.json())
	return JSONResponse(vars(container_info))


@app.get('/health/')
async def is_container_alive(id: int):
	container_info = manager.container_status(id)
	return JSONResponse(vars(container_info))


@app.get('/data/')
async def get_container_data(id: int):
	container_info = manager.get_container_data(id)
	return JSONResponse(vars(container_info))


@app.get('/stop/')
async def stop_container(id: int):
	container_info = manager.stop_container(id)
	return JSONResponse(vars(container_info))


@app.get('/data/tensorboard/')
async def get_tensorboard_link(id: int):
	tb_link = manager.start_tensorboard(id)
	return JSONResponse(vars(tb_link))


@app.get('/remove/')
async def remove_container(id: int):
	container_info = manager.remove_container(id)
	return JSONResponse(vars(container_info))


@app.get('/kill/')
async def kill_container(id: int):
	# TODO: evaluate if this is needed/useful
	container_info = manager.kill_container(id)
	return JSONResponse(vars(container_info))
