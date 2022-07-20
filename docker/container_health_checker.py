import logging
import os
import subprocess
import time
from datetime import datetime, timedelta

import psutil
from container_db_manager import ContainerDB
from docker_manager import DockerManager

path_to_log_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_files')
if not os.path.isdir(path_to_log_files):
	os.makedirs(path_to_log_files)

logging.basicConfig(filename=os.path.join(path_to_log_files, 'container_health_checker.log'),
				filemode='a',
				format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
				datefmt='%H:%M:%S',
				level=logging.DEBUG)

docker_manager_logger = logging.getLogger('docker-manager')
manager = DockerManager(docker_manager_logger)
container_db = ContainerDB()
last_docker_info = None
diff = timedelta(minutes=5)
last_time = datetime.now()
print('successfully started container health checker, waiting for container to die')


def get_system_information():
	global last_time
	cpu = psutil.cpu_percent(percpu=True)
	ram = psutil.virtual_memory()
	io = psutil.disk_io_counters()
	gpu = subprocess.check_output('nvidia-smi', shell=True)
	container_db.update_system(cpu, ram, io, gpu)
	last_time = datetime.now()


while True:
	if manager.check_for_running_recommerce_container():
		try:
			is_exited, docker_info = manager.check_health_of_all_container()
			if is_exited and last_docker_info != docker_info:
				last_docker_info = docker_info
				polished_data = [item[1:-1].split(',') for item in docker_info.status.split(';')]
				polished_data = [(container_id[1:-1].strip(), exit_code.strip()) for container_id, exit_code in polished_data]
				container_db.they_are_exited(polished_data)
		except Exception as e:
			print(f'something went wrong {e}')
		# get memory, cpu and io information
		current_time = datetime.now()
		if current_time - last_time > diff:
			get_system_information()
	time.sleep(5)
