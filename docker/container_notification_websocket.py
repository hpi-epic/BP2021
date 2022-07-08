import asyncio
import json
import logging

import uvicorn
from docker_manager import DockerManager
from fastapi import FastAPI, WebSocket

from docker.utils import setup_logging

setup_logging('websocket')


class ConnectionManager:
	def __init__(self):
		self.active_connections = []

	async def connect(self, websocket: WebSocket):
		await websocket.accept()
		self.active_connections.append(websocket)
		logging.info(f'got new connection of {websocket}, current connections: {self.active_connections}')

	def disconnect(self, websocket: WebSocket):
		self.active_connections.remove(websocket)

	async def broadcast(self, message: str):
		for connection in self.active_connections:
			await connection.send_json(message)


manager = DockerManager()
connection_manager = ConnectionManager()
app = FastAPI()


@app.websocket('/wss')
async def websocket_endpoint(websocket: WebSocket):
	await connection_manager.connect(websocket)
	last_docker_info = None
	try:
		while True:
			await asyncio.sleep(5)
			is_exited, docker_info = manager.check_health_of_all_container()
			if is_exited and last_docker_info != docker_info:
				logging.info(f'Sending information about stopped container {docker_info}')
				await connection_manager.broadcast(json.dumps(vars(docker_info)))
				last_docker_info = docker_info
	except Exception:
		connection_manager.disconnect(websocket)


if __name__ == '__main__':
	uvicorn.run('container_notification_websocket:app',
		host='0.0.0.0',
		port=8001,
		ssl_keyfile='/etc/sslzertifikat/api_cert.key',
		ssl_certfile='/etc/sslzertifikat/api_cert.crt')
