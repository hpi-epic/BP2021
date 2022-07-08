import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(name, level=logging.DEBUG):
	log_file_dir = './log_files'
	if not os.path.exists(log_file_dir):
		os.makedirs(log_file_dir)
	logging.basicConfig(
		handlers=[RotatingFileHandler(f'{log_file_dir}/{name}.log', maxBytes=100000, backupCount=10)],
		level=level,
		format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
		datefmt='%Y-%m-%dT%H:%M:%S')


class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
