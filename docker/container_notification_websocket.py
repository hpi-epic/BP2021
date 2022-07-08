import asyncio
import json
import logging
import os

import uvicorn
from docker_manager import DockerManager
from fastapi import FastAPI, WebSocket

logger = logging.getLogger('uvicorn.error')
path_to_log_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_files')
if not os.path.isdir(path_to_log_files):
	os.makedirs(path_to_log_files)


class ConnectionManager:
	def __init__(self):
		self.active_connections = []

	async def connect(self, websocket: WebSocket):
		await websocket.accept()
		self.active_connections.append(websocket)
		logger.info(f'got new connection of {websocket}, current connections: {self.active_connections}')

	def disconnect(self, websocket: WebSocket):
		self.active_connections.remove(websocket)

	async def broadcast(self, message: str):
		for connection in self.active_connections:
			await connection.send_json(message)


manager = DockerManager(logger)
connection_manager = ConnectionManager()
app = FastAPI()


@app.on_event('startup')
async def startup_event():
	logger.info('started websocket')


@app.websocket('/wss')
async def websocket_endpoint(websocket: WebSocket):
	await connection_manager.connect(websocket)
	last_docker_info = None
	try:
		while True:
			await asyncio.sleep(5)
			is_exited, docker_info = manager.check_health_of_all_container()
			if is_exited and last_docker_info != docker_info:
				logger.info(f'Sending information about stopped container {docker_info}')
				await connection_manager.broadcast(json.dumps(vars(docker_info)))
				last_docker_info = docker_info
	except Exception:
		connection_manager.disconnect(websocket)


if __name__ == '__main__':
	uvicorn.run('container_notification_websocket:app',
		host='0.0.0.0',
		port=8001,
		log_config='./log_websocket.ini',
		ssl_keyfile='/etc/sslzertifikat/api_cert.key',
		ssl_certfile='/etc/sslzertifikat/api_cert.crt')
