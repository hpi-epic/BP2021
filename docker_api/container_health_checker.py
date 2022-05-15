import time

import psutil
from container_db_manager import ContainerDB
from docker_manager import DockerManager

manager = DockerManager()
container_db = ContainerDB()
last_docker_info = None
print('successfully started container health checker, waiting for container to die')
while True:
	is_exited, docker_info = manager.check_health_of_all_container()
	if is_exited and last_docker_info != docker_info:
		last_docker_info = docker_info
		polished_data = [item[1:-1].split(',') for item in docker_info.status.split(';')]
		polished_data = [(container_id[1:-1].strip(), exit_code.strip()) for container_id, exit_code in polished_data]
		container_db.they_are_exited(polished_data)
	# get memory, cpu and io information
	print('cpu', psutil.cpu_percent())
	print('RAM', psutil.virtual_memory())
	print('percet used RAM', psutil.virtual_memory().percent)
	print('percent avail memory', psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
	print('io', psutil.disk_io_counters())
	print()
	print()
	print()
	time.sleep(5)
