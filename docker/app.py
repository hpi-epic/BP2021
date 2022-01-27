# app.py
from fake_docker_manager import AlphaBusinessDockerManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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


@app.get('/data/')
async def get_container_data(id: int):
	container_info = manager.get_container_data(id)
	return JSONResponse(vars(container_info))


@app.get('/kill/')
async def kill_container(id: int):
	container_info = manager.kill_container(id)
	return JSONResponse(vars(container_info))
