# app.py
from fake_docker_manager import AlphaBusinessDockerManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# This file should expose a RESTful api for using the docker container with the following routes:
# POST /start/<docker_id>
# GET /health/<docker_id>
# GET /data/<docker_id>
# GET /data/tensorboard/<docker_id>
# GET /kill/<docker_id>

# start API with uvicorn app:app --reload
manager = AlphaBusinessDockerManager()

app = FastAPI()


@app.post('/start')
async def start_container(config: Request):
	container_info = manager.start_docker(await config.json())
	return JSONResponse(vars(container_info))


@app.get('/health/')
async def is_container_alive(id: int):
	container_info = manager.is_container_alive(id)
	return JSONResponse(vars(container_info))


@app.get('/data/tensorboard')
async def get_tensorboard_link(id: int):
	container_info = manager.get_tensorboard_link(id)
	return JSONResponse(vars(container_info))


@app.get('/kill/')
async def kill_container(id: int):
	container_info = manager.kill_container(id)
	return JSONResponse(vars(container_info))


def iterfile():
	with open('logo.tar', mode='rb') as file_like:
		yield from file_like


@app.get('/data')
async def data(id: int):
	return StreamingResponse(iterfile(), media_type='application/x-tar')
