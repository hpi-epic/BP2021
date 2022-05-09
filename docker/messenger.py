import os
from datetime import datetime
from time import sleep

import requests
from utils import bcolors


def get_telegram_bot_credentials() -> tuple:
	bot_token = None
	bot_chatID = None
	try:
		with open('.env.txt', 'r') as file:
			data = file.readlines()
			bot_token = data[2].strip()
			bot_chatID = data[3].strip()
	except Exception as e:
		print(f'{bcolors.FAIL}Could not get bot_token and bot_chatID: {e}{bcolors.ENDC}')
	return bot_token, bot_chatID


def telegram_bot_sendtext(bot_message) -> bool:
	bot_token, bot_chatID = get_telegram_bot_credentials()
	send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={bot_chatID}&text={bot_message}'
	response = requests.get(send_text)
	if not response.ok:
		print(f'{bcolors.WARNING}{response}{bcolors.ENDC}')
		return False
	return True


def telegram_bot_send_document(path_to_file) -> bool:
	bot_token, bot_chatID = get_telegram_bot_credentials()
	document = {'document': (os.path.dirname(path_to_file), open(path_to_file, 'rb'))}
	response = requests.post(f'https://api.telegram.org/bot{bot_token}/sendDocument?chat_id={bot_chatID}', files=document)
	if not response.ok:
		print(f'{bcolors.WARNING}{response}{bcolors.ENDC}')
		return False
	return True


def read_files_in_logs_dir(log_file_dir: str) -> dict:
	all_log_file_names = [f for f in os.listdir(log_file_dir) if os.path.isfile(os.path.join(log_file_dir, f))]
	result = {}
	for log_file_name in all_log_file_names:
		try:
			with open(os.path.join(log_file_dir, log_file_name), 'r') as log_file:
				data = log_file.readlines()
				result[log_file_name] = [line.strip() for line in data]
		except Exception as e:
			print(f'{bcolors.WARNING}Could not open {log_file_name}, due to {e}{bcolors.ENDC}')
	return result


log_file_dir = './log_files'
last_log_files = read_files_in_logs_dir(log_file_dir)
last_send = datetime.now()
new_errors = []
notification_interval = 600  # in seconds
print(f'{bcolors.OKGREEN}{last_send}successfully started messenger')
print('waiting for logs in {log_file_dir} to change and notify you{bcolors.ENDC}')
while True:
	current_log_files = read_files_in_logs_dir(log_file_dir)
	for file_name in last_log_files.keys():
		if not os.path.exists(os.path.join(log_file_dir, file_name)):
			print(f'[{datetime.now}]\t{bcolors.WARNING}file {file_name} does not exist anymore{bcolors.ENDC}')
		if last_log_files[file_name] != current_log_files[file_name]:
			print(f'[{datetime.now}]\t{bcolors.OKCYAN}{file_name} has changed{bcolors.ENDC}')
			new_errors += [list(set(current_log_files[file_name]) - set(last_log_files[file_name]))]
	last_log_files = current_log_files
	if new_errors:
		diff = (datetime.now() - last_send).total_seconds()
		print(f'New errors are here, sending them in {notification_interval - diff}')
		if diff > notification_interval:
			print(f'[{datetime.now}]\t{bcolors.OKGREEN}send telegram {bcolors.ENDC}')
			current_time = datetime.now()
			errors_as_one_list = ['\n'.join(x) for x in new_errors]
			if telegram_bot_sendtext(f'error log of {current_time}\n' + '\n'.join(errors_as_one_list)):
				new_errors = []
				last_send = current_time
	sleep(5)