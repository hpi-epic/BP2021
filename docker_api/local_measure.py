import os
from datetime import datetime
from uuid import uuid4

RESULT_FILE_NAME = 'times.csv'

num_container = 1

c_id = uuid4()
with open(RESULT_FILE_NAME, 'a') as file:
	file.write(f'{c_id};{num_container};{datetime.now()};start\n')
os.system('recommerce -c training')
with open(RESULT_FILE_NAME, 'a') as file:
	file.write(f'{c_id};{num_container};{datetime.now()};end\n')
