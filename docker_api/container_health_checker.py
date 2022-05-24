import logging
import time
from datetime import datetime, timedelta

import psutil
from container_db_manager import ContainerDB
from docker_manager import DockerManager

from docker_api.utils import setup_logging

manager = DockerManager()
container_db = ContainerDB()
last_docker_info = None
diff = timedelta(minutes=5)
last_time = datetime.now()
setup_logging('health_checker')
print('successfully started container health checker, waiting for container to die')


def get_system_information():
	global last_time
	cpu = psutil.cpu_percent(percpu=True)
	ram = psutil.virtual_memory()
	io = psutil.disk_io_counters()
	import nvidia_smi
	nvidia_smi.nvmlInit()
	deviceCount = nvidia_smi.nvmlDeviceGetCount()
	for i in range(deviceCount):
		handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
		info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
		print('Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)'.format(i, nvidia_smi.nvmlDeviceGetName(handle),
			100*info.free/info.total, info.total, info.free, info.used))
	nvidia_smi.nvmlShutdown()
	gpu = ''
	container_db.update_system(cpu, ram, io, gpu)
	last_time = datetime.now()


while True:
	try:
		is_exited, docker_info = manager.check_health_of_all_container()
		if is_exited and last_docker_info != docker_info:
			last_docker_info = docker_info
			polished_data = [item[1:-1].split(',') for item in docker_info.status.split(';')]
			polished_data = [(container_id[1:-1].strip(), exit_code.strip()) for container_id, exit_code in polished_data]
			container_db.they_are_exited(polished_data)
	except Exception as e:
		logging.warning(f'something went wrong {e}')
	# get memory, cpu and io information
	current_time = datetime.now()
	if current_time - last_time > diff:
		get_system_information()
	time.sleep(5)
