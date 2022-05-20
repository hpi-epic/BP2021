import time
from datetime import datetime, timedelta

import psutil

RESULT_FILE = 'system.csv'
diff = timedelta(minutes=5)
last_time = datetime.now()


def get_system_information():
	global last_time
	cpu = psutil.cpu_percent(percpu=True)
	ram = psutil.virtual_memory()
	io = psutil.disk_io_counters()
	last_time = datetime.now()
	with open(RESULT_FILE, 'a') as file:
		file.write(f'{datetime.now()};{str(cpu)};{str(ram)};{str(io)}\n')


while True:
	# get memory, cpu and io information
	current_time = datetime.now()
	if current_time - last_time > diff:
		get_system_information()
	time.sleep(5)
