import asyncio
import json

import uvicorn
from docker_manager import DockerManager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


class ConnectionManager:
	def __init__(self):
		self.active_connections = []

	async def connect(self, websocket: WebSocket):
		await websocket.accept()
		self.active_connections.append(websocket)

	def disconnect(self, websocket: WebSocket):
		self.active_connections.remove(websocket)


manager = DockerManager()
connection_manager = ConnectionManager()
app = FastAPI()


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
	await connection_manager.connect(websocket)
	last_docker_info = None
	try:
		while True:
			await asyncio.sleep(1)
			is_exited, docker_info = manager.check_health_of_all_container()
			if is_exited and last_docker_info != docker_info:
				await websocket.send_json(json.dumps(vars(docker_info)))
				last_docker_info = docker_info
	except WebSocketDisconnect:
		connection_manager.disconnect(websocket)


if __name__ == '__main__':
	uvicorn.run('container_notification_websocket:app', host='0.0.0.0', port=8001)
