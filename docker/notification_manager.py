import datetime
import os

from utils import bcolors


class NotificationManager():
	def __init__(self, log_file_name: str) -> None:
		self.path_to_logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_files', log_file_name + '.log')
		try:
			with open(self.path_to_logfile, 'w') as file:
				file.write(f'[{datetime.datetime.now()}]\tStarted logging\n')
			print(f'{bcolors.OKGREEN}Successfully started notification manager{bcolors.ENDC}')
		except Exception as e:
			print(f'{bcolors.FAIL}Failed to instantiate notification manager, because {e}{bcolors.ENDC}')
		self.current_line = 1
		self.error_cache = []

	def error(self, error_msg: str) -> None:
		try:
			with open(self.path_to_logfile, 'a') as file:
				for line in self.error_cache:
					file.write(line)
					self.error_cache.remove(line)
				file.write(f'[{datetime.datetime.now()}]\t{error_msg}\n')
		except Exception as e:
			self.error_cache += [f'[{datetime.datetime.now()}]\t{error_msg}\n']
			print(f'{bcolors.WARNING}Failed to write message ({error_msg}) because {e}{bcolors.ENDC}')

	def get_errors(self) -> list:
		try:
			with open(self.path_to_logfile, 'a') as file:
				for line in self.error_cache:
					file.write(line)
					self.error_cache.remove(line)
			with open(self.path_to_logfile, 'r') as file:
				return file.readlines()
		except Exception as e:
			print(f'{bcolors.WARNING}Failed to read file because {e}{bcolors.ENDC}')
# import requests

# def telegram_bot_sendtext(bot_message):

# 	bot_token = '5327123109:AAENRZ0IQ9X7Xdy-HQR540fmu3D4VSZS2Qs'
# 	bot_chatID = '-781418860'
# 	send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' +
#   bot_chatID + '&parse_mode=Markdown&text=' + bot_message

# 	#response = requests.get(send_text)
# 	with open('./test.py', 'rb') as file:
# 		a = file.read()
# 	files = {'files': open('test.py','rb')}
# 	values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}
# 	response = requests.post('https://api.telegram.org/bot' + bot_token + '/sendDocument', files=files, data=values)

# 	return response.json()

# test = telegram_bot_sendtext("Testing Telegram bot")
# print(test)
