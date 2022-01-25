# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fake_docker_manager import AlphaBusinessDockerManager
from pydantic import BaseModel

# This file should expose a RESTful api for using the docker container with the following routes:
# POST /start/<docker_id>
# GET /health/<docker_id>
# GET /data/<docker_id>
# GET /data/<docker_id>/tensorboard
# GET /kill/<docker_id>

manager = AlphaBusinessDockerManager()

app = FastAPI()

@app.post("/start", status_code=201)
async def start_container(config: Request):
	container_info = manager.start_docker(await config.json())
	return JSONResponse(vars(container_info))


@app.get("/health/")
async def is_container_alive(id: int):
	container_info = manager.is_container_alive(id)
	return JSONResponse(vars(container_info))


@app.get("/data/")
async def is_container_alive(id: int):
	container_info = manager.get_container_data(id)
	return JSONResponse(vars(container_info))