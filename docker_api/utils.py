import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(name):
	log_file_dir = './log_files'
	if not os.path.exists(log_file_dir):
		os.makedirs(log_file_dir)
	logging.basicConfig(
		handlers=[RotatingFileHandler(f'{log_file_dir}/{name}.log', maxBytes=100000, backupCount=10)],
		level=logging.DEBUG,
		format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
		datefmt='%Y-%m-%dT%H:%M:%S')
