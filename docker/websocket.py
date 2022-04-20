# app.py

import uvicorn
from docker_manager import DockerManager
from fastapi import FastAPI, WebSocket

manager = DockerManager()

app = FastAPI()


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()
	while True:
		is_exited, exited_container_ids = manager.check_health_of_all_container()
		if is_exited:
			await websocket.send_text(exited_container_ids)


if __name__ == '__main__':
	uvicorn.run('app:app', host='0.0.0.0', port=8000)
