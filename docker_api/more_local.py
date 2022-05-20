import os
import time

for i in range(10):
	print(f'###############################START: {i}###############################')
	cmd_command = 'python3 ./local_measure & '
	os.system((cmd_command * i)[:-2].strip())
	print(f'+++++++++++++++++++++++++++++++++++++++DONE: {i}+++++++++++++++++++++++++++++++++++++++')
	time.sleep(305)
